"""Core classes."""

import dataclasses
import logging

from enum import Enum, auto
from random import choice, sample
from typing import Iterable, List, Optional, Set, Tuple

from no_thanks.elo import calculate_multiplayer_elo_rating_update
from no_thanks.utils import pairwise

LOGGER = logging.getLogger(__name__)


class Action(Enum):
    """Possible actions."""

    TAKE = auto()
    PASS = auto()


ACTIONS = tuple(Action)


@dataclasses.dataclass
class GameState:
    """The state of the game from one player's perspective."""

    number_of_opponents: int
    cards_in_draw_pile: int
    current_card: int

    tokens_on_card: int
    tokens_in_hand_of_this_player: int

    card_m3_in_front_of_this_player: bool
    card_m2_in_front_of_this_player: bool
    card_m1_in_front_of_this_player: bool
    card_p1_in_front_of_this_player: bool
    card_p2_in_front_of_this_player: bool
    card_p3_in_front_of_this_player: bool

    card_m3_in_front_of_other_players: bool
    card_m2_in_front_of_other_players: bool
    card_m1_in_front_of_other_players: bool
    card_p1_in_front_of_other_players: bool
    card_p2_in_front_of_other_players: bool
    card_p3_in_front_of_other_players: bool


class Game:
    """A game of No Thanks!"""

    NUM_PLAYERS_MIN = 3
    NUM_PLAYERS_MAX = 7
    CARD_MIN = 3
    CARD_MAX = 35
    NUM_CARDS_DISCARD = 9

    players: Tuple["Player", ...]
    draw_pile: List[int]
    current_player: int
    tokens_on_card: int

    def __init__(self, players: Iterable["Player"]) -> None:
        self.players = tuple(players)
        assert self.NUM_PLAYERS_MIN <= len(self.players) <= self.NUM_PLAYERS_MAX
        self.reset()

    def reset(self) -> None:
        """Reset the game."""

        self.current_player = 0
        self.tokens_on_card = 0
        self.players = tuple(sample(population=self.players, k=len(self.players)))

        tokens_per_player = min(55 // len(self.players), 11)
        for player in self.players:
            player.reset(game=self, tokens=tokens_per_player)

        self.draw_pile = sample(
            population=range(self.CARD_MIN, self.CARD_MAX + 1),
            k=self.CARD_MAX - self.NUM_CARDS_DISCARD,
        )

        LOGGER.debug("Draw pile: %s", ", ".join(map(str, self.draw_pile)))

    @property
    def finished(self) -> bool:
        """Has this game finished?"""
        return not self.draw_pile

    @property
    def sort_players(self) -> Tuple["Player", ...]:
        """Sort players by their score."""
        attr = "score" if self.finished else "score_cards"
        return tuple(sorted(self.players, key=lambda player: -getattr(player, attr)))

    def state(self, player: "Player") -> GameState:
        """Get the state of the game from the perspective of a player."""

        assert not self.finished, "Game must not be finished"
        assert self.draw_pile, "Game must not be finished"
        assert (
            player in self.players
        ), f"Player <{player.name}> does not take part in this game"

        current_card = self.draw_pile[0]

        return GameState(
            number_of_opponents=len(self.players) - 1,
            cards_in_draw_pile=len(self.draw_pile) - 1,
            current_card=current_card,
            tokens_on_card=self.tokens_on_card,
            tokens_in_hand_of_this_player=player.tokens,
            card_m3_in_front_of_this_player=current_card - 3 in player.cards,
            card_m2_in_front_of_this_player=current_card - 2 in player.cards,
            card_m1_in_front_of_this_player=current_card - 1 in player.cards,
            card_p1_in_front_of_this_player=current_card + 1 in player.cards,
            card_p2_in_front_of_this_player=current_card + 2 in player.cards,
            card_p3_in_front_of_this_player=current_card + 3 in player.cards,
            card_m3_in_front_of_other_players=any(
                current_card - 3 in p.cards for p in self.players if p is not player
            ),
            card_m2_in_front_of_other_players=any(
                current_card - 2 in p.cards for p in self.players if p is not player
            ),
            card_m1_in_front_of_other_players=any(
                current_card - 1 in p.cards for p in self.players if p is not player
            ),
            card_p1_in_front_of_other_players=any(
                current_card + 1 in p.cards for p in self.players if p is not player
            ),
            card_p2_in_front_of_other_players=any(
                current_card + 2 in p.cards for p in self.players if p is not player
            ),
            card_p3_in_front_of_other_players=any(
                current_card + 3 in p.cards for p in self.players if p is not player
            ),
        )

    def play(self) -> Tuple["Player", ...]:
        """Play the game."""

        LOGGER.info("Starting a game with %d players", len(self.players))

        while not self.finished:
            player = self.players[self.current_player]
            LOGGER.debug(
                "Card: %d; token(s): %d; card(s) remaining: %d, active player: %s",
                self.draw_pile[0],
                self.tokens_on_card,
                len(self.draw_pile) - 1,
                player,
            )

            action = player.action()
            assert action is Action.TAKE or player.tokens > 0
            LOGGER.debug("%s chose action: %s", player, action)

            if action is Action.TAKE:
                player.cards.add(self.draw_pile.pop(0))
                player.tokens += self.tokens_on_card
                LOGGER.debug(
                    "%s now has %d token(s) and runs: %s",
                    player,
                    player.tokens,
                    player.runs,
                )
                self.tokens_on_card = 0

            else:
                player.tokens -= 1
                self.tokens_on_card += 1
                self.current_player += 1
                while self.current_player >= len(self.players):
                    self.current_player -= len(self.players)

        LOGGER.info("Game has finished, final results:")

        results = self.sort_players

        for pos, player in enumerate(results):
            LOGGER.info(
                "#%d %s with %d points: %d token(s) and %s",
                pos + 1,
                player,
                player.score,
                player.tokens,
                player.runs,
            )

        return results

    def update_elo_ratings(self) -> None:
        """Update the ELO ratings of the players."""

        assert self.finished, "Game must be finished before updating ELO ratings"

        players = self.sort_players
        elo_ratings = tuple(player.elo_rating for player in players)
        scores = tuple(player.score for player in players)

        updates = calculate_multiplayer_elo_rating_update(
            elo_ratings=elo_ratings,
            scores=scores,
        )

        new_elo_ratings = tuple(
            elo_rating + update for elo_rating, update in zip(elo_ratings, updates)
        )

        LOGGER.info(
            "ELO ratings before game: %s; after game: %s",
            elo_ratings,
            new_elo_ratings,
        )

        for player, elo_rating in zip(players, new_elo_ratings):
            player.elo_rating = elo_rating


def _make_runs(cards: Iterable[int]) -> Iterable[Iterable[int]]:
    cards = sorted(cards)
    if not cards:
        return
    run = [cards[0]]
    for prev, succ in pairwise(cards):
        if succ == prev + 1:
            run.append(succ)
        else:
            yield run
            run = [succ]
    yield run


class Player:
    """A player."""

    name: str
    elo_rating: float

    game: Game
    tokens: int
    cards: Set[int]

    def __init__(self, name: str, elo_rating: Optional[float] = None) -> None:
        self.name = name
        self.elo_rating = elo_rating or 1200

    def __str__(self) -> str:
        return f"Player <{self.name}>"

    def reset(self, game: Game, tokens: int) -> None:
        """Reset the player."""
        self.game = game
        self.tokens = tokens
        self.cards = set()

    def action(self) -> Action:
        """Choose an action."""
        return Action.TAKE if self.tokens <= 0 else choice(ACTIONS)

    @property
    def runs(self) -> Tuple[Tuple[int, ...], ...]:
        """The sequencial runs formed by this player's cards."""
        return tuple(tuple(run) for run in _make_runs(self.cards))

    @property
    def score_cards(self) -> int:
        """Minus points incurred from cards."""
        return sum(-run[0] for run in self.runs)

    @property
    def score(self) -> int:
        """Total score."""
        return self.score_cards + self.tokens
