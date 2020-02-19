# -*- coding: utf-8 -*-

"""Players."""

import inquirer

from .core import Action, Player


class Human(Player):
    """Human interactive input."""

    def action(self: "Human") -> Action:
        """Prompt for an action."""

        action_tags = [
            inquirer.questions.TaggedValue(label=str(a), value=a) for a in Action
        ]
        question = inquirer.List(
            name="action",
            message="Choose an action",
            choices=action_tags,
            carousel=True,
        )
        answer = inquirer.prompt([question])

        assert answer
        assert isinstance(answer.get("action"), Action)

        return answer["action"]
