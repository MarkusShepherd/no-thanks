"""Elo rating system implementation."""

import itertools
import math
from typing import Iterable, Tuple


def calculate_expected_outcome(elo_rating_1: float, elo_rating_2: float) -> float:
    """
    Calculate the expected score of player 1 in a match against player 2.
    """
    return 1 / (1 + math.pow(10, (elo_rating_2 - elo_rating_1) / 400))


def calculate_elo_rating_update(
    elo_rating_1: float,
    elo_rating_2: float,
    score_1: float,
    score_2: float,
    k_factor: float = 32,
) -> float:
    """
    Calculate the Elo rating points player 1 gains in a match against player 2.
    """

    expected_outcome = calculate_expected_outcome(elo_rating_1, elo_rating_2)
    actual_outcome = 1 if score_1 > score_2 else 0 if score_1 < score_2 else 0.5

    return k_factor * (actual_outcome - expected_outcome)


def calculate_multiplayer_elo_rating_update(
    elo_ratings: Iterable[float],
    scores: Iterable[float],
    k_factor: float = 32,
) -> Tuple[float, ...]:
    """
    Calculate the Elo rating points each player gains in a multiplayer match.
    """

    elo_ratings = tuple(elo_ratings)
    scores = tuple(scores)
    assert len(elo_ratings) == len(scores), "Number of players and scores must be equal"

    updates = [0.0 for _ in elo_ratings]

    for player_1, player_2 in itertools.combinations(range(len(elo_ratings)), 2):
        update = calculate_elo_rating_update(
            elo_ratings[player_1],
            elo_ratings[player_2],
            scores[player_1],
            scores[player_2],
            k_factor,
        )
        updates[player_1] += update
        updates[player_2] -= update

    return tuple(updates)
