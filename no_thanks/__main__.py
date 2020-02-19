# -*- coding: utf-8 -*-

"""CLI entry point."""

import argparse
import logging
import sys

from typing import List

from .core import Game, Player
from .players import Heuristic, Human

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No Thanks!")
    parser.add_argument(
        "names", nargs="*", default=("You",), help="names of human players"
    )
    parser.add_argument(
        "--players", "-p", type=int, default=5, help="number of players in total"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="log level (repeat for more verbosity)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(levelname)-4.4s [%(name)s:%(lineno)s] %(message)s",
    )

    LOGGER.info(args)

    names = args.names or ()
    num_players = max(args.players, len(names))
    assert Game.NUM_PLAYERS_MIN <= num_players <= Game.NUM_PLAYERS_MAX

    players: List[Player] = [Human(name=name) for name in names]
    players += [Heuristic(name=f"AI #{i + 1}") for i in range(num_players - len(names))]

    game = Game(players=players)
    game.play()


if __name__ == "__main__":
    main()
