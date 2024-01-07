"""Players."""

import logging
import random


import inquirer

from no_thanks.core import Action, Player

LOGGER = logging.getLogger(__name__)


class HumanPlayer(Player):
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


class HeuristicPlayer(Player):
    """Use heuristics to choose actions."""

    def action(self) -> Action:
        """Choose an action based on heuristics."""
        proba = self.take_proba()
        LOGGER.debug("Probability of %s: %.1f%%", Action.TAKE, 100 * proba)
        return (
            Action.TAKE if self.tokens <= 0 or random.random() <= proba else Action.PASS
        )

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
