"""Policy gradient algorithm."""

import dataclasses
import logging
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, List, Optional, Tuple, Union
import numpy as np

import torch
from torch import nn
from torch import optim
import tqdm

from no_thanks.core import Action, Game, GameState
from no_thanks.players import HeuristicPlayer
from no_thanks.utils import pairwise

LOGGER = logging.getLogger(__name__)
UNSAFE_CHARACTERS = re.compile(r"\W+")


class PolicyNetwork(nn.Module):
    """Policy network."""

    def __init__(self, hidden_layers: Tuple[int, ...]):
        super().__init__()
        features = (len(dataclasses.fields(GameState)),) + hidden_layers + (1,)
        self.linear_layers = nn.ModuleList(
            nn.Linear(i, o) for i, o in pairwise(features)
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        for layer in self.linear_layers[:-1]:
            x = self.relu(layer(x))
        x = self.linear_layers[-1](x)
        x = self.sigmoid(x)
        return x


class PolicyGradientPlayer(HeuristicPlayer):
    """Use policy gradient to choose actions."""

    HIDDEN_LAYERS = (32, 16, 8)

    states: List[np.ndarray[Any, np.dtype[np.int8]]]
    actions: List[int]

    def __init__(
        self,
        name: str,
        policy_net: PolicyNetwork,
        *,
        discount_factor: float = 0.99,
        learning_rate: float = 0.01,
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
        self.states = []
        self.actions = []

    def action(self) -> Action:
        """Choose an action based on policy gradient."""
        state = self.game.state(self)
        self.states.append(state.to_array())
        action = super().action()
        self.actions.append(action.value)
        return action

    def take_proba(self) -> float:
        """Probability to play TAKE."""
        state = self.game.state(self)
        state_tensor = torch.FloatTensor(state.to_array())
        proba = self.policy_net(state_tensor)
        return proba[0]

    def update_weights(self, reward: float) -> None:
        """Update the weights of the policy network."""

        if not self.states or not self.actions:
            LOGGER.info(
                "No states or actions recorded, unable to update weights of <%s>",
                self.name,
            )
            return

        assert self.states, "No states recorded"
        assert self.actions, "No actions recorded"
        assert len(self.actions) == len(self.states), "States and actions mismatch"

        self.optimizer.zero_grad()

        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(
            [
                self.discount_factor**i * reward
                for i in range(len(self.states) - 1, -1, -1)
            ]
        )

        action_probs = self.policy_net(states).squeeze()
        log_probs = torch.log(
            action_probs * actions + (1 - action_probs) * (1 - actions)
        )

        baseline = torch.mean(rewards)
        advantage = rewards - baseline
        policy_loss = -torch.sum(log_probs * advantage)

        policy_loss.backward()
        self.optimizer.step()


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

    if (save_frequency % num_games) != 0:
        trainer.save_players(save_dir=save_dir, overwrite=True)

    players = sorted(trainer.players, key=lambda p: p.elo_rating, reverse=True)
    for player in players:
        print(f"{player.name}: {int(player.elo_rating)}")
        print(player.policy_net.state_dict())


if __name__ == "__main__":
    main()
