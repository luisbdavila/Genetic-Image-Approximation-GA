"""
Parent selection strategies for the genetic algorithm.

Three strategies are implemented, all accepting lower-is-better RMSE values:
  - Tournament : sample k individuals, keep the best one.  The default.
  - Ranking    : assign selection probabilities based on rank order.
  - Roulette   : assign selection probabilities inversely proportional to fitness.

Tournament is the default because it is cheap, robust to tightly clustered
fitness values, and its tournament size k gives direct control over selection
pressure.
"""

import numpy as np

from .. import population

# Type alias used throughout the module
Individual = list[population.Triangle]


_VALID_SELECTION_TYPES = {"tournament", "ranking", "roulette"}


def normalize_selection_type(selection_type: str) -> str:
    """Normalise a strategy name to 'tournament', 'ranking', or 'roulette'."""
    normalized = selection_type.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "roulette_wheel":
        normalized = "roulette"
    if normalized not in _VALID_SELECTION_TYPES:
        raise ValueError(
            "selection_type must be one of: tournament, ranking, roulette."
        )
    return normalized


def select_parent(
    population_data: list[Individual],
    fitness_values: list[float],
    selection_type: str,
    tournament_size: int = 3,
) -> Individual:
    """Select and return one parent using the configured strategy."""

    if not population_data:
        raise ValueError("population_data must not be empty.")
    if len(population_data) != len(fitness_values):
        raise ValueError(
            "population_data and fitness_values must have the same length."
        )

    normalized_type = normalize_selection_type(selection_type)

    if normalized_type == "tournament":
        return tournament_selection(
            population_data,
            fitness_values,
            tournament_size=tournament_size,
        )
    if normalized_type == "ranking":
        return ranking_selection(population_data, fitness_values)

    # Default fallback: roulette wheel
    return roulette_wheel_selection(population_data, fitness_values)


def tournament_selection(
    population_data: list[Individual],
    fitness_values: list[float],
    tournament_size: int = 3,
) -> Individual:
    """
    Sample tournament_size individuals and return the one with the lowest RMSE.

    Larger tournament sizes increase selection pressure; smaller sizes preserve
    more diversity.
    """

    if tournament_size <= 0:
        raise ValueError("tournament_size must be positive.")

    replace = len(population_data) < tournament_size
    candidate_indices = np.random.choice(
        len(population_data), size=tournament_size, replace=replace
    )
    best_index = min(candidate_indices, key=lambda index: fitness_values[int(index)])

    return population_data[int(best_index)]


def ranking_selection(
    population_data: list[Individual],
    fitness_values: list[float],
) -> Individual:
    """
    Sample one individual with probability proportional to its rank.

    Rank 1 (best) gets weight N, rank N (worst) gets weight 1.
    More robust than roulette when fitness values cluster tightly.
    """

    ranked_indices = np.argsort(fitness_values)
    weights        = np.arange(len(ranked_indices), 0, -1, dtype=np.float64)
    probabilities  = weights / weights.sum()
    selected_position = int(np.random.choice(len(ranked_indices), p=probabilities))

    return population_data[int(ranked_indices[selected_position])]


def roulette_wheel_selection(
    population_data: list[Individual],
    fitness_values: list[float],
) -> Individual:
    """
    Sample one individual with probability proportional to 1 / (fitness + 1).

    Fitness is inverted because we minimise RMSE (lower = better).
    Subtracting the minimum before inversion keeps weights well-conditioned.
    """

    fitness_array  = np.asarray(fitness_values, dtype=np.float64)
    shifted        = fitness_array - np.min(fitness_array)
    weights        = 1.0 / (shifted + 1.0)
    probabilities  = weights / weights.sum()

    selected_index = int(np.random.choice(len(population_data), p=probabilities))

    return population_data[selected_index]
