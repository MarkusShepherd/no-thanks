"""CLI entry point."""

import argparse
import logging
from pathlib import Path
import pickle
import random
import sys

from typing import List, Optional, Union

from .core import Game, Player
from .players import Heuristic, Human, ParametricHeuristic

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No Thanks!")
    parser.add_argument(
        "names",
        nargs="*",
        default=("You",),
        help="names of human players",
    )
    parser.add_argument(
        "--players",
        "-p",
        type=int,
        default=5,
        help="number of players in total",
    )
    parser.add_argument(
        "--strategies-dir",
        "-s",
        default=Path(__file__).parent.parent / "trained_strategies" / "genetic",
        help="stagies directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="log level (repeat for more verbosity)",
    )
    return parser.parse_args()


def load_strategies(
    save_dir: Union[str, Path],
    num_strategies: int,
    top_strategies: Optional[int] = None,
) -> List[Heuristic]:
    """Load strategies from disk."""

    save_dir = Path(save_dir).resolve()
    file_paths = sorted(save_dir.glob("*.pickle")) if save_dir.is_dir() else ()

    if not file_paths:
        LOGGER.warning(
            "Directory <%s> does not exist or is empty, using simple heuristics instead",
            save_dir,
        )
        return [
            Heuristic(name=f"AI #{i + 1}")
            if random.random() < 0.5
            else ParametricHeuristic.random_weights(name=f"PAI #{i + 1}")
            for i in range(num_strategies)
        ]

    LOGGER.info("Loading %d strategies from <%s>", num_strategies, save_dir)

    if top_strategies is not None:
        file_paths = file_paths[:top_strategies]

    sampled_paths = (
        random.sample(file_paths, num_strategies)
        if len(file_paths) > num_strategies
        else random.choices(file_paths, k=num_strategies)
    )

    return [pickle.load(file.open("rb")) for file in sampled_paths]


def main() -> None:
    """CLI entry point."""

    args = _parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="%(levelname)-4.4s [%(name)s:%(lineno)s] %(message)s",
    )

    LOGGER.info(args)

    names = args.names or ()
    num_players = max(args.players, len(names))
    assert Game.NUM_PLAYERS_MIN <= num_players <= Game.NUM_PLAYERS_MAX

    players: List[Player] = [Human(name=name) for name in names]
    players += load_strategies(
        save_dir=args.strategies_dir,
        num_strategies=num_players - len(players),
        top_strategies=2 * num_players,
    )

    game = Game(players=players)
    game.play()


if __name__ == "__main__":
    main()
