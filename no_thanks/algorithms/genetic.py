"""Train a strategy using genetic algorithms."""

import random
from typing import Optional, Tuple

from no_thanks.core import Game
from no_thanks.players import ParametricHeuristic


class GeneticTrainer:
    """Train a strategy using genetic algorithms."""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 1000,
        inheritance_rate: float = 0.7,
        mutation_rate: float = 0.1,
    ):
        """Initialize the trainer."""

        self.population_size = population_size
        self.generations = generations
        self.inheritance_rate = inheritance_rate
        self.mutation_rate = mutation_rate

        self.current_generation = 0
        self.current_population_count = 0
        self.population: Optional[Tuple[ParametricHeuristic, ...]] = None

    def reset(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Reset the trainer."""
        self.current_generation = 0
        self.population = tuple(
            ParametricHeuristic.random_weights(
                name=f"AI #{i:05d} (Gen #00000)",
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
        results = game.play()

        return game
