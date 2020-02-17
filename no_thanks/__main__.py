# -*- coding: utf-8 -*-

"""CLI entry point."""

import logging
import sys

from .core import Game, Player


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(levelname)-4.4s [%(name)s:%(lineno)s] %(message)s",
    )

    num_players = 5
    players = (Player(name=str(i + 1)) for i in range(num_players))
    game = Game(players=players)
    game.play()


if __name__ == "__main__":
    main()
