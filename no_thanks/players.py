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
        LOGGER.info("Probability to take: %.3f", proba)
        return Action.TAKE if self.tokens <= 0 or random() <= proba else Action.PASS

    def take_proba(self) -> float:
        """Probability to play TAKE."""

        card = self.game.draw_pile[0]
        tokens_on_card = self.game.tokens_on_card
        value = card - tokens_on_card - 1

        if self.tokens <= 0 or value <= 0:
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

    def take_proba(self) -> float:
        """Probability to play TAKE."""
        # TODO: implement using parameters
        return super().take_proba()
