"""Train a strategy using genetic algorithms."""

import logging
import random
import sys
from typing import Optional, Tuple

import tqdm

from no_thanks.core import Game
from no_thanks.players import ParametricHeuristic

LOGGER = logging.getLogger(__name__)


class GeneticTrainer:
    """Train a strategy using genetic algorithms."""

    def __init__(
        self,
        population_size: int = 100,
        games_per_generation: int = 1000,
        generations: int = 10000,
        inheritance_rate: float = 0.7,
        reproduction_rate: float = 0.25,
        mutation_rate: float = 0.05,
    ):
        """Initialize the trainer."""

        self.population_size = population_size
        self.games_per_generation = games_per_generation
        self.generations = generations
        self.inheritance_rate = inheritance_rate
        self.reproduction_rate = reproduction_rate
        self.mutation_rate = mutation_rate

        assert 0 <= self.inheritance_rate <= 1
        assert 0 <= self.reproduction_rate <= 1
        assert 0 <= self.inheritance_rate + self.reproduction_rate <= 1
        assert 0 <= self.mutation_rate <= 1

        self.current_generation = 0
        self.current_population_count = 0
        self.population: Optional[Tuple[ParametricHeuristic, ...]] = None

    def reset(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Reset the trainer."""
        self.current_generation = 0
        self.population = tuple(
            ParametricHeuristic.random_weights(
                name=f"AI #{i:05d} (gen #00000)",
                mean=mean,
                std=std,
            )
            for i in range(self.population_size)
        )
        self.current_population_count = self.population_size

    def play_game(
        self,
        *,
        min_players: Optional[int] = None,
        max_players: Optional[int] = None,
    ) -> Game:
        """Play a game with a sample of the current population."""

        assert self.population is not None, "Population not initialized"

        min_players = min_players or Game.NUM_PLAYERS_MIN
        max_players = max_players or Game.NUM_PLAYERS_MAX

        assert min_players <= max_players

        num_players = (
            min_players
            if min_players == max_players
            else random.randint(min_players, max_players)
        )
        players = random.sample(self.population, num_players)

        game = Game(players)
        game.play()
        game.update_elo_ratings()

        return game

    def play_generation(self, evolve_population: bool = True) -> None:
        """Play a generation."""

        assert self.population is not None, "Population not initialized"

        self.current_generation += 1

        LOGGER.warning("Playing generation #%05d", self.current_generation)

        # TODO: Parallelize this
        for _ in tqdm.trange(self.games_per_generation):
            self.play_game()

        self.population = tuple(
            sorted(
                self.population,
                key=lambda player: player.elo_rating,
                reverse=True,
            )
        )

        LOGGER.warning(
            "Best player: %s (Elo: %d)",
            self.population[0].name,
            self.population[0].elo_rating,
        )
        LOGGER.warning(
            "Worst player: %s (Elo: %d)",
            self.population[-1].name,
            self.population[-1].elo_rating,
        )
        LOGGER.warning("Finished generation #%05d", self.current_generation)

        if not evolve_population:
            return

        num_inheritance = int(self.population_size * self.inheritance_rate)
        num_reproduction = int(self.population_size * self.reproduction_rate)
        num_new = self.population_size - num_inheritance - num_reproduction
        assert num_new >= 0

        LOGGER.info(
            "Evolution for next generation: %d inheritance, %d reproduction, %d new",
            num_inheritance,
            num_reproduction,
            num_new,
        )

        population_inheritance = self.population[:num_inheritance]

        population_reproduction = tuple(
            ParametricHeuristic.mate(
                *random.sample(population_inheritance, 2),
                name=f"AI #{self.current_population_count + i + 1:05d} "
                + f"(gen #{self.current_generation:05d}, child)",
            )
            for i in range(num_reproduction)
        )
        self.current_population_count += num_reproduction

        population_new = tuple(
            ParametricHeuristic.random_weights(
                name=f"AI #{self.current_population_count + i + 1:05d} "
                + f"(gen #{self.current_generation:05d}, new)",
                mean=0,
                std=1,
            )
            for i in range(num_new)
        )
        self.current_population_count += num_new

        self.population = (
            population_inheritance + population_reproduction + population_new
        )

        for player in self.population:
            player.elo_rating = 1200
            if random.random() < self.mutation_rate:
                player.mutate()
                player.name += f" [mutated gen #{self.current_generation:05d}]"

    def train(self) -> None:
        """Train a strategy."""

        assert self.population is not None, "Population not initialized"

        for i in range(self.generations):
            last_generation = i == self.generations - 1
            self.play_generation(evolve_population=not last_generation)


def main():
    """Run the trainer."""

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)-4.4s [%(name)s:%(lineno)s] %(message)s",
    )

    trainer = GeneticTrainer(generations=10)
    trainer.reset()
    trainer.train()

    for player in trainer.population[:3]:
        print(player.name, player.elo_rating)


if __name__ == "__main__":
    main()
