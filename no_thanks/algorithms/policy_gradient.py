from typing import Any, List, Optional
import numpy as np

import torch
from torch import nn
from torch import optim

from no_thanks.core import Action, Game
from no_thanks.players import HeuristicPlayer


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
        return self.policy_net(torch.FloatTensor(state.to_array()))

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
