# -*- coding: utf-8 -*-

"""Players."""

import logging

from random import random

import inquirer

from .core import Action, Player

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


class ParameterHeuristic(Heuristic):
    """Use heuristics with parameters to choose actions."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # TODO: parameters, e.g.,
        # * cards in draw pile
        # * tokens in hand
        # * tokens on card
        # * card value
        # * ±1/2/3 card in front of this player
        # * ±1/2/3 card in front of other players
        # * number of players / expected value on next turn

    def take_proba(self) -> float:
        """Probability to play TAKE."""

        number_of_opponents = len(self.game.players) - 1
        cards_in_draw_pile = len(self.game.draw_pile)
        current_card = self.game.draw_pile[0]
        tokens_on_card = self.game.tokens_on_card
        tokens_in_hand = self.tokens
        current_value = current_card - tokens_on_card - 1
        future_value = current_value - number_of_opponents

        cards_in_front_of_this_player = {
            f"current_{i:+d}": current_card + i in self.cards
            for i in (-3, -2, -1, +1, +2, +3)
        }
        cards_in_front_of_other_players = {
            f"current_{i:+d}": any(
                current_card + i in opponent.cards
                for opponent in self.game.players
                if opponent is not self
            )
            for i in (-3, -2, -1, +1, +2, +3)
        }

        if (tokens_in_hand <= 0) or (current_value <= 0):
            return 1

        if (
            cards_in_front_of_this_player["current_+1"]
            or cards_in_front_of_this_player["current_-1"]
        ) and (future_value <= 0):
            return 1

        return super().take_proba()
