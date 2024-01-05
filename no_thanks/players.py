"""Players."""

import dataclasses
import logging
import random

from statistics import NormalDist
from typing import Dict, Optional

import inquirer

from no_thanks.core import Action, Player
from no_thanks.utils import sigmoid

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class StrategyWeights:
    """Weights for a strategy."""

    current_card_weight: float = 0.0
    current_value_weight: float = 0.0
    future_value_weight: float = 0.0
    tokens_in_hand_weight: float = 0.0
    tokens_on_card_weight: float = 0.0
    cards_in_draw_pile_weight: float = 0.0
    number_of_opponents_weight: float = 0.0
    cards_in_front_of_this_player_weight: Dict[int, float]
    cards_in_front_of_other_players_weight: Dict[int, float]


class ParametricHeuristic(Heuristic):
    """Use heuristics with parameters to choose actions."""

    CARD_DISTANCES = (-3, -2, -1, +1, +2, +3)

    @classmethod
    def random_weights(
        cls,
        name: str,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> "ParametricHeuristic":
        """Create a heuristic with random parameters."""

        dist = NormalDist(mean, std)

        (
            current_card_weight,
            current_value_weight,
            future_value_weight,
            tokens_in_hand_weight,
            tokens_on_card_weight,
            cards_in_draw_pile_weight,
            number_of_opponents_weight,
        ) = dist.samples(7)

        cards_in_front_of_this_player_weight = dict(
            zip(
                cls.CARD_DISTANCES,
                dist.samples(len(cls.CARD_DISTANCES)),
            )
        )
        cards_in_front_of_other_players_weight = dict(
            zip(
                cls.CARD_DISTANCES,
                dist.samples(len(cls.CARD_DISTANCES)),
            )
        )

        strategy_weights = StrategyWeights(
            current_card_weight=current_card_weight,
            current_value_weight=current_value_weight,
            future_value_weight=future_value_weight,
            tokens_in_hand_weight=tokens_in_hand_weight,
            tokens_on_card_weight=tokens_on_card_weight,
            cards_in_draw_pile_weight=cards_in_draw_pile_weight,
            number_of_opponents_weight=number_of_opponents_weight,
            cards_in_front_of_this_player_weight=cards_in_front_of_this_player_weight,
            cards_in_front_of_other_players_weight=cards_in_front_of_other_players_weight,
        )

        return cls(
            name=name,
            strategy_weights=strategy_weights,
        )

    def __init__(
        self,
        name: str,
        strategy_weights: StrategyWeights,
        *,
        elo_rating: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, elo_rating=elo_rating)
        self.strategy_weights = strategy_weights

        for i in self.CARD_DISTANCES:
            self.strategy_weights.cards_in_front_of_this_player_weight.setdefault(
                i, 0.0
            )
            self.strategy_weights.cards_in_front_of_other_players_weight.setdefault(
                i, 0.0
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
            d: current_card + d in self.cards for d in self.CARD_DISTANCES
        }
        cards_in_front_of_other_players = {
            d: any(
                current_card + d in opponent.cards
                for opponent in self.game.players
                if opponent is not self
            )
            for d in self.CARD_DISTANCES
        }

        # if (tokens_in_hand <= 0) or (current_value <= 0):
        #     return 1

        # if (
        #     cards_in_front_of_this_player[1]
        #     or cards_in_front_of_this_player[-1]
        # ) and (future_value <= 0):
        #     return 1

        logit = (
            self.strategy_weights.current_card_weight * current_card
            + self.strategy_weights.current_value_weight * current_value
            + self.strategy_weights.future_value_weight * future_value
            + self.strategy_weights.tokens_in_hand_weight * tokens_in_hand
            + self.strategy_weights.tokens_on_card_weight * tokens_on_card
            + self.strategy_weights.cards_in_draw_pile_weight * cards_in_draw_pile
            + self.strategy_weights.number_of_opponents_weight * number_of_opponents
            + sum(
                self.strategy_weights.cards_in_front_of_this_player_weight[k] * v
                for k, v in cards_in_front_of_this_player.items()
            )
            + sum(
                self.strategy_weights.cards_in_front_of_other_players_weight[k] * v
                for k, v in cards_in_front_of_other_players.items()
            )
        )

        return sigmoid(logit)

    def mutate(self, mean: float = 0.0, std: float = 1.0) -> "ParametricHeuristic":
        """Randomly mutate on of the weight parameters."""

        dict_attrs = (
            "cards_in_front_of_this_player_weight",
            "cards_in_front_of_other_players_weight",
        )
        attrs = (
            "current_card_weight",
            "current_value_weight",
            "future_value_weight",
            "tokens_in_hand_weight",
            "tokens_on_card_weight",
            "cards_in_draw_pile_weight",
            "number_of_opponents_weight",
        ) + dict_attrs

        attr = random.choice(attrs)
        new_value = NormalDist(mean, std).samples(1)[0]

        if attr in dict_attrs:
            distance = random.choice(self.CARD_DISTANCES)
            getattr(self.strategy_weights, attr)[distance] = new_value
        else:
            self.strategy_weights = dataclasses.replace(
                self.strategy_weights,
                **{attr: new_value},  # type: ignore[arg-type]
            )

        return self

    @classmethod
    def mate(cls, *parents: "ParametricHeuristic", name: str) -> "ParametricHeuristic":
        """Mate two (or more) parents to create a child."""

        attrs = (
            "current_card_weight",
            "current_value_weight",
            "future_value_weight",
            "tokens_in_hand_weight",
            "tokens_on_card_weight",
            "cards_in_draw_pile_weight",
            "number_of_opponents_weight",
        )
        kwargs = {
            attr: sum(getattr(parent.strategy_weights, attr) for parent in parents)
            / len(parents)
            for attr in attrs
        }

        kwargs["cards_in_front_of_this_player_weight"] = {
            d: sum(
                parent.strategy_weights.cards_in_front_of_this_player_weight[d]
                for parent in parents
            )
            / len(parents)
            for d in cls.CARD_DISTANCES
        }
        kwargs["cards_in_front_of_other_players_weight"] = {
            d: sum(
                parent.strategy_weights.cards_in_front_of_other_players_weight[d]
                for parent in parents
            )
            / len(parents)
            for d in cls.CARD_DISTANCES
        }

        strategy_weights = StrategyWeights(**kwargs)

        return cls(name=name, strategy_weights=strategy_weights)
