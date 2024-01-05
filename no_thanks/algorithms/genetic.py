"""Train a strategy using genetic algorithms."""

from ..players import ParametricHeuristic


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

        self.population = tuple(
            ParametricHeuristic.random_weights(
                name=f"AI #{i:05d}",
                mean=0.0,
                std=10.0,
            )
            for i in range(self.population_size)
        )
