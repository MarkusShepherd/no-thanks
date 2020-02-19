# -*- coding: utf-8 -*-

"""Core classes."""

import logging

from enum import Enum, auto
from random import choice, sample
from typing import Iterable, List, Set, Tuple

from .utils import pairwise

LOGGER = logging.getLogger(__name__)


class Action(Enum):
    """Possible actions."""

    TAKE = auto()
    PASS = auto()


ACTIONS = tuple(Action)


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

    def __init__(self: "Game", players: Iterable["Player"]) -> None:
        self.players = tuple(players)
        assert self.NUM_PLAYERS_MIN <= len(self.players) <= self.NUM_PLAYERS_MAX
        self.reset()

    def reset(self: "Game") -> None:
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

        LOGGER.info("Draw pile: %s", ", ".join(map(str, self.draw_pile)))

    @property
    def finished(self: "Game") -> bool:
        """Has this game finished?"""
        return not self.draw_pile

    @property
    def sort_players(self: "Game") -> Tuple["Player", ...]:
        """Sort players by their score."""
        attr = "score" if self.finished else "score_cards"
        return tuple(sorted(self.players, key=lambda player: -getattr(player, attr)))

    def play(self: "Game") -> Tuple["Player", ...]:
        """Play the game."""

        LOGGER.info("Starting a game with %d players", len(self.players))

        while not self.finished:
            player = self.players[self.current_player]
            LOGGER.info(
                "Card: %d; token(s): %d; card(s) remaining: %d, active player: %s",
                self.draw_pile[0],
                self.tokens_on_card,
                len(self.draw_pile) - 1,
                player,
            )

            action = player.action()
            assert action is Action.TAKE or player.tokens > 0
            LOGGER.info("%s chose action: %s", player, action)

            if action is Action.TAKE:
                player.cards.add(self.draw_pile.pop(0))
                player.tokens += self.tokens_on_card
                LOGGER.info(
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
    game: Game
    tokens: int
    cards: Set[int]

    def __init__(self: "Player", name: str) -> None:
        self.name = name

    def __str__(self: "Player") -> str:
        return f"Player <{self.name}>"

    def reset(self: "Player", game: Game, tokens: int) -> None:
        """Reset the player."""
        self.game = game
        self.tokens = tokens
        self.cards = set()

    def action(self: "Player") -> Action:
        """Choose an action."""
        return Action.TAKE if self.tokens <= 0 else choice(ACTIONS)

    @property
    def runs(self: "Player") -> Tuple[Tuple[int, ...], ...]:
        """The sequencial runs formed by this player's cards."""
        return tuple(tuple(run) for run in _make_runs(self.cards))

    @property
    def score_cards(self: "Player") -> int:
        """Minus points incurred from cards."""
        return sum(-run[0] for run in self.runs)

    @property
    def score(self: "Player") -> int:
        """Total score."""
        return self.score_cards + self.tokens
