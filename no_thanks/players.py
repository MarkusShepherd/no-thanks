# -*- coding: utf-8 -*-

"""Players."""

import logging

from random import random
from typing import Dict

import inquirer

from .core import Action, Player
from .utils import sigmoid

LOGGER = logging.getLogger(__name__)


class Human(Player):
    """Human interactive input."""

    def action(self) -> Action:
        """Prompt for an action."""

        actions = [Action.TAKE] if self.tokens <= 0 else reversed(Action)
        action_tags = [
            inquirer.questions.TaggedValue(label=str(a), value=a) for a in actions
        ]
        question = inquirer.List(
            name="action",
            message=f"Choose an action [{self.tokens} token(s) in hand; runs: {self.runs}]",
            choices=action_tags,
            carousel=True,
        )
        answer = inquirer.prompt([question])

        assert answer
        assert isinstance(answer.get("action"), Action)

        return answer["action"]


class Heuristic(Player):
    """Use heuristics to choose actions."""

    def action(self) -> Action:
        """Choose an action based on heuristics."""
        proba = self.take_proba()
        LOGGER.info("Probability of %s: %.1f%%", Action.TAKE, 100 * proba)
        return Action.TAKE if self.tokens <= 0 or random() <= proba else Action.PASS

    def take_proba(self) -> float:
        """Probability to play TAKE."""

        card = self.game.draw_pile[0]
        tokens_on_card = self.game.tokens_on_card
        value = card - tokens_on_card - 1

        if (self.tokens <= 0) or (value <= 0):
            return 1

        proba = 0.05 if value >= 20 else 1 - value / 20

        proba *= 1 + 1 / self.tokens

        if (card - 1 in self.cards) or (card + 1 in self.cards):
            proba *= 2

        return max(min(proba, 1), 0)


class ParametricHeuristic(Heuristic):
    """Use heuristics with parameters to choose actions."""

    def __init__(
        self,
        name: str,
        current_card_weight: float,
        current_value_weight: float,
        future_value_weight: float,
        tokens_in_hand_weight: float,
        tokens_on_card_weight: float,
        cards_in_draw_pile_weight: float,
        number_of_opponents_weight: float,
        cards_in_front_of_this_player_weight: Dict[int, float],
        cards_in_front_of_other_players_weight: Dict[int, float],
    ) -> None:
        super().__init__(name=name)
        self.current_card_weight = current_card_weight
        self.current_value_weight = current_value_weight
        self.future_value_weight = future_value_weight
        self.tokens_in_hand_weight = tokens_in_hand_weight
        self.tokens_on_card_weight = tokens_on_card_weight
        self.cards_in_draw_pile_weight = cards_in_draw_pile_weight
        self.number_of_opponents_weight = number_of_opponents_weight
        self.cards_in_front_of_this_player_weight = cards_in_front_of_this_player_weight
        self.cards_in_front_of_other_players_weight = (
            cards_in_front_of_other_players_weight
        )

    def take_proba(self) -> float:
        """Probability to play TAKE depending on chose parameters."""

        number_of_opponents = len(self.game.players) - 1
        cards_in_draw_pile = len(self.game.draw_pile)
        current_card = self.game.draw_pile[0]
        tokens_on_card = self.game.tokens_on_card
        tokens_in_hand = self.tokens
        current_value = current_card - tokens_on_card - 1
        future_value = current_value - number_of_opponents

        cards_in_front_of_this_player = {
            i: current_card + i in self.cards for i in (-3, -2, -1, +1, +2, +3)
        }
        cards_in_front_of_other_players = {
            i: any(
                current_card + i in opponent.cards
                for opponent in self.game.players
                if opponent is not self
            )
            for i in (-3, -2, -1, +1, +2, +3)
        }

        # if (tokens_in_hand <= 0) or (current_value <= 0):
        #     return 1

        # if (
        #     cards_in_front_of_this_player[1]
        #     or cards_in_front_of_this_player[-1]
        # ) and (future_value <= 0):
        #     return 1

        logit = (
            self.current_card_weight * current_card
            + self.current_value_weight * current_value
            + self.future_value_weight * future_value
            + self.tokens_in_hand_weight * tokens_in_hand
            + self.tokens_on_card_weight * tokens_on_card
            + self.cards_in_draw_pile_weight * cards_in_draw_pile
            + self.number_of_opponents_weight * number_of_opponents
            + sum(
                self.cards_in_front_of_this_player_weight[k] * v
                for k, v in cards_in_front_of_this_player.items()
            )
            + sum(
                self.cards_in_front_of_other_players_weight[k] * v
                for k, v in cards_in_front_of_other_players.items()
            )
        )

        return sigmoid(logit)
