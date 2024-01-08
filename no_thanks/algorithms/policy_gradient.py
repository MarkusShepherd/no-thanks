"""Policy gradient algorithm."""

import dataclasses
from pathlib import Path
import random
from typing import Any, List, Optional, Tuple, Union
import numpy as np

import torch
from torch import nn
from torch import optim

from no_thanks.core import Action, Game, GameState
from no_thanks.players import HeuristicPlayer


class PolicyNetwork(nn.Module):
    """Policy network."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(len(dataclasses.fields(GameState)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class PolicyGradientPlayer(HeuristicPlayer):
    """Use policy gradient to choose actions."""

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
        print(state_tensor.shape, state_tensor)
        proba = self.policy_net(state_tensor)
        print(proba.shape, proba)
        return proba

    def update_weights(self, reward: float) -> None:
        """Update the weights of the policy network."""

        assert self.states, "No states recorded"
        assert self.actions, "No actions recorded"
        assert len(self.actions) == len(self.states), "States and actions mismatch"

        self.optimizer.zero_grad()

        states = torch.FloatTensor(self.states)
        print(states.shape, states)
        actions = torch.LongTensor(self.actions)
        print(actions.shape, actions)
        rewards = torch.FloatTensor(
            [
                self.discount_factor**i * reward
                for i in range(len(self.states) - 1, -1, -1)
            ]
        )
        print(rewards.shape, rewards)

        action_probs = self.policy_net(states)
        print(action_probs.shape, action_probs)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        print(log_probs.shape, log_probs)

        baseline = torch.mean(rewards)
        advantage = rewards - baseline
        policy_loss = -torch.sum(log_probs * advantage)
        print(policy_loss.shape, policy_loss)

        policy_loss.backward()
        self.optimizer.step()


class PolicyGradientTrainer:
    """Train a policy gradient player."""

    current_game_num: int
    players: Tuple[PolicyGradientPlayer, ...]

    def __init__(self, num_games: int = 1_000):
        self.num_games = num_games

    def reset(self) -> None:
        """Reset the trainer."""
        self.current_game_num = 0
        self.players = tuple(
            PolicyGradientPlayer(
                name=f"PG #{i:03d}",
                policy_net=PolicyNetwork(),
            )
            for i in range(Game.NUM_PLAYERS_MAX)
        )

    def resume(self, save_dir: Union[str, Path]) -> None:
        """Resume training."""
        raise NotImplementedError

    def play_game(
        self,
        *,
        min_players: Optional[int] = None,
        max_players: Optional[int] = None,
    ) -> Game:
        """Play a game."""

        print(f"Playing game {self.current_game_num + 1} of {self.num_games}…")

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

        for player, reward in zip(self.players, rewards):
            player.update_weights(reward)

        self.current_game_num += 1

        return game

    def train(self) -> None:
        """Train policy gradient players."""

        for _ in range(self.num_games):
            self.play_game()
