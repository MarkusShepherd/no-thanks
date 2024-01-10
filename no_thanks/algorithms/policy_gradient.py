"""Policy gradient algorithm."""

import dataclasses
import logging
from pathlib import Path
import random
import re
import shutil
import sys
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Categorical
import tqdm

from no_thanks.core import Action, Game, GameState, Player
from no_thanks.utils import pairwise

LOGGER = logging.getLogger(__name__)
UNSAFE_CHARACTERS = re.compile(r"\W+")


class PolicyNetwork(nn.Module):
    """Policy network."""

    def __init__(self, hidden_layers: Tuple[int, ...]):
        super().__init__()
        features = (len(dataclasses.fields(GameState)),) + hidden_layers + (2,)
        self.linear_layers = nn.ModuleList(
            nn.Linear(i, o) for i, o in pairwise(features)
        )

    def forward(self, x):
        """Forward pass."""
        for layer in self.linear_layers[:-1]:
            x = F.relu(layer(x))
        x = self.linear_layers[-1](x)
        x = F.softmax(x, dim=-1)
        return x


class PolicyGradientPlayer(Player):
    """Use policy gradient to choose actions."""

    HIDDEN_LAYERS = (32, 16, 8)

    log_probas: List[torch.Tensor]

    def __init__(
        self,
        name: str,
        policy_net: PolicyNetwork,
        *,
        discount_factor: float = 0.9,
        learning_rate: float = 0.00001,
        elo_rating: Optional[float] = None,
    ):
        super().__init__(name, elo_rating=elo_rating)
        self.policy_net = policy_net
        self.discount_factor = discount_factor
        self.optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    def save(self, path: Union[str, Path]) -> None:
        """Save the player."""
        path = Path(path).resolve()
        LOGGER.info("Saving player <%s> to <%s>", self.name, path)
        torch.save(self.policy_net.state_dict(), path)

    @classmethod
    def load(cls, name: str, path: Union[str, Path]) -> "PolicyGradientPlayer":
        """Load a player."""
        path = Path(path).resolve()
        LOGGER.info("Loading player from <%s>", path)
        policy_net = PolicyNetwork(cls.HIDDEN_LAYERS)
        policy_net.load_state_dict(torch.load(path))
        return cls(name=name, policy_net=policy_net)

    def reset(self, game: Game, tokens: int) -> None:
        """Reset the player."""
        super().reset(game, tokens)
        self.log_probas = []

    def action(self) -> Action:
        """Choose an action based on policy gradient."""
        if self.tokens <= 0:
            # TODO: should we give a negative reward
            # to nudge the policy towards taking cards
            # when we have no tokens in hand?
            return Action.TAKE
        state = self.game.state(self).to_array()
        state_tensor = torch.FloatTensor(state)
        proba = self.policy_net(state_tensor)

        distro = Categorical(proba)
        action = distro.sample()
        self.log_probas.append(torch.log(proba[action]))
        LOGGER.debug(
            "Probability of %s: %.1f%%; actual action: %s",
            Action.TAKE,
            100 * proba[1],
            Action(action.item()),
        )

        return Action(action.item())

    def update_weights(self, reward: float) -> None:
        """Update the weights of the policy network."""

        if not self.log_probas:
            LOGGER.info(
                "No states or actions recorded, unable to update weights of <%s>",
                self.name,
            )
            return

        assert self.log_probas, "No log probabilities recorded"

        log_probas = torch.stack(self.log_probas)
        discounted_rewards = torch.FloatTensor(
            [
                self.discount_factor**i * reward
                for i in reversed(range(len(log_probas)))
            ]
        )

        # TODO: use a baseline?
        policy_loss = -torch.sum(log_probas * discounted_rewards)

        policy_loss.backward()
        # TODO: torch.nn.utils.clip_grad_norm_?
        self.optimizer.step()
        self.optimizer.zero_grad()


class PolicyGradientTrainer:
    """Train a policy gradient player."""

    current_game_num: int
    players: Tuple[PolicyGradientPlayer, ...]

    def __init__(self, num_games: int):
        self.num_games = num_games

    def save_players(
        self,
        save_dir: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """Save the players."""

        save_dir = Path(save_dir).resolve()
        LOGGER.info("Saving players to <%s>", save_dir)

        if save_dir.exists():
            if overwrite:
                shutil.rmtree(save_dir)
            else:
                raise FileExistsError(f"Directory <{save_dir}> already exists")
        save_dir.mkdir(parents=True, exist_ok=True)

        players = sorted(self.players, key=lambda p: p.elo_rating, reverse=True)

        for i, player in enumerate(players):
            sanitized_name = UNSAFE_CHARACTERS.sub("_", player.name)
            file_name = f"{i + 1:05d}_{sanitized_name}.pt"
            file_path = save_dir / file_name
            player.save(file_path)

    def reset(self) -> None:
        """Reset the trainer."""
        self.current_game_num = 0
        self.players = tuple(
            PolicyGradientPlayer(
                name=f"PG #{i:03d}",
                policy_net=PolicyNetwork(PolicyGradientPlayer.HIDDEN_LAYERS),
            )
            for i in range(Game.NUM_PLAYERS_MAX)
        )

    def resume(self, save_dir: Union[str, Path]) -> None:
        """Resume training."""
        save_dir = Path(save_dir).resolve()
        LOGGER.info("Loading players from <%s>", save_dir)
        self.players = (
            tuple(
                PolicyGradientPlayer.load(name=f"PG #{i:03d} [resumed]", path=path)
                for i, path in enumerate(sorted(save_dir.glob("*.pt")))
            )
            if save_dir.exists()
            else ()
        )
        self.players += tuple(
            PolicyGradientPlayer(
                name=f"PG #{i:03d}",
                policy_net=PolicyNetwork(PolicyGradientPlayer.HIDDEN_LAYERS),
            )
            for i in range(len(self.players), Game.NUM_PLAYERS_MAX)
        )
        self.current_game_num = 0

    def play_game(
        self,
        *,
        min_players: Optional[int] = None,
        max_players: Optional[int] = None,
    ) -> Game:
        """Play a game."""

        min_players = min_players or Game.NUM_PLAYERS_MIN
        max_players = max_players or Game.NUM_PLAYERS_MAX

        assert (
            Game.NUM_PLAYERS_MIN <= min_players <= max_players <= Game.NUM_PLAYERS_MAX
        )

        num_players = (
            min_players
            if min_players == max_players
            else random.randint(min_players, max_players)
        )
        players = random.sample(self.players, num_players)

        game = Game(players=players)
        game.play()
        game.update_elo_ratings()

        winning_score = max(player.score for player in players)
        # TODO: should we increase the penalty for losing by the margin?
        rewards = [-1 if player.score < winning_score else +1 for player in players]

        for player, reward in zip(players, rewards):
            player.update_weights(reward)

        self.current_game_num += 1

        return game

    def train(
        self,
        save_dir: Union[None, str, Path] = None,
        save_frequency: int = 1_000,
    ) -> None:
        """Train policy gradient players."""

        for i in tqdm.trange(self.num_games):
            self.play_game()
            if save_dir and ((i + 1) % save_frequency == 0):
                self.save_players(save_dir, overwrite=True)


def main():
    """Main function."""

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)-4.4s [%(name)s:%(lineno)s] %(message)s",
    )

    save_dir = (
        Path(__file__).parent.parent.parent / "trained_strategies" / "policy_gradient"
    )

    num_games = 1_000_000
    save_frequency = 1000

    trainer = PolicyGradientTrainer(num_games=num_games)
    trainer.resume(save_dir=save_dir)
    trainer.train(save_dir=save_dir, save_frequency=save_frequency)

    if (num_games % save_frequency) != 0:
        trainer.save_players(save_dir=save_dir, overwrite=True)

    players = sorted(trainer.players, key=lambda p: p.elo_rating, reverse=True)
    for player in players:
        print(f"{player.name}: {int(player.elo_rating)}")
        print(player.policy_net.state_dict())


if __name__ == "__main__":
    main()
