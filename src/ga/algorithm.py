"""
Core GeneticAlgorithm class for triangle-based image approximation.

How the GA works
----------------
Each candidate solution (individual) is a list of Triangle objects.
When rendered, those triangles form a coloured image that the algorithm
tries to make look as close as possible to a target photo.

One generation of the GA:
  1. Evaluate   — render every individual and compute its RMSE vs. the target.
  2. Select     — pick pairs of parents from the population, biased toward
                  individuals with lower RMSE.
  3. Crossover  — combine each parent pair to produce one or more children.
  4. Mutate     — randomly perturb some children to maintain diversity.
  5. Replace    — the children become the next generation's population
                  (optionally keeping the best individuals unchanged via elitism).

Optional features
-----------------
  Elitism           : copy the N best individuals unchanged into each new
                      generation so the best solution never gets worse.
  Adaptive mutation : automatically raise the mutation rate when the search
                      stagnates, then lower it again as progress resumes.
  Random immigrants : inject a few fully random individuals each generation
                      to maintain diversity and help escape local optima.
  Seeded init       : initialise triangle colours from random target pixels
                      rather than fully random colours for a better start.
  Progress output   : print a one-line summary after each generation.
  Parallel eval     : use threads or processes to speed up fitness evaluation.
"""

import copy
import time
from collections.abc import Callable

import numpy as np

from .. import population
from . import evaluation, logs, selection

# ---------------------------------------------------------------------------
# Type aliases — give descriptive names to complex callable signatures
# ---------------------------------------------------------------------------

Individual = list[population.Triangle]

# A fitness function takes two RGB arrays (target, generated) → scalar
FitnessFunction = evaluation.FitnessFunction

# A crossover result can be one individual, a tuple, or a list of individuals
CrossoverResult = Individual | tuple[Individual, ...] | list[Individual]

# Crossover function signature: (parent1, parent2, rate) → child(ren)
CrossoverFunction = Callable[[Individual, Individual, float], CrossoverResult]

# Mutation function signature: (individual, rate, width, height, alpha_range) → individual
MutationFunction = Callable[
    [Individual, float, int, int, population.AlphaRange], Individual
]

# Union type for validation helpers
OperatorFunction = CrossoverFunction | MutationFunction

# Re-export for callers that import from this module
EvaluationBackend = evaluation.EvaluationBackend

# Progress callback receives the GenerationLog dict each generation
ProgressCallback = Callable[[logs.GenerationLog], None]


class GeneticAlgorithm:
    """
    Evolves a population of triangle-based image candidates toward lower RMSE.

    Each individual is a complete candidate image composed of ``n_triangles``
    triangles.  The algorithm maintains ``population_size`` individuals per
    generation, evaluates their rendered images against the target, and
    iteratively improves them through selection, crossover, and mutation.

    Usage example::

        ga = GeneticAlgorithm(
            target=target_array,
            fitness_function=fitness.compute_rmse,
            population_size=100,
            generations=200,
            crossover_function=cross_over.single_point_crossover,
            crossover_rate=0.9,
            mutation_function=mutate.random_triangle_mutation,
            mutation_rate=0.1,
            elitism=2,
            selection_type="tournament",
        )
        best_rmse, history = ga.run()
        best_image = ga.best_individual
    """

    # -----------------------------------------------------------------------
    # Private static helpers (validation / counting)
    # -----------------------------------------------------------------------

    @staticmethod
    def _resolve_operator_rate(
        rate_name: str,
        rate: float | None,
        operator: OperatorFunction | None,
    ) -> float:
        """Return 0.0 if no operator; validate and return the rate otherwise."""

        if rate is None:
            if operator is not None:
                raise ValueError(
                    f"{rate_name} must be provided when its function is set."
                )
            return 0.0

        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{rate_name} must be between 0 and 1.")

        return float(rate)

    @staticmethod
    def _normalize_initial_population(
        initial_population: list[Individual] | None,
        n_triangles: int,
    ) -> list[Individual] | None:
        """Validate and deep-copy an external initial population, or return None."""

        if initial_population is None:
            return None
        if not initial_population:
            raise ValueError("initial_population must not be empty when provided.")

        # Deep copy so that mutations during the run do not affect the caller's data
        copied_population = copy.deepcopy(initial_population)
        for individual in copied_population:
            if len(individual) != n_triangles:
                raise ValueError(
                    "Each initial_population individual must have exactly "
                    "n_triangles triangles."
                )

        return copied_population

    @staticmethod
    def _count_changed_triangles(
        before_mutation: Individual,
        after_mutation: Individual,
    ) -> int:
        """Count how many triangles differ between before and after mutation."""

        if len(before_mutation) != len(after_mutation):
            # Length mismatch — treat all as changed
            return max(len(before_mutation), len(after_mutation))

        return sum(
            1
            for before_triangle, after_triangle in zip(
                before_mutation, after_mutation, strict=True
            )
            if before_triangle != after_triangle
        )

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(
        self,
        target: np.ndarray,
        fitness_function: FitnessFunction,
        population_size: int,
        generations: int,
        crossover_rate: float | None = None,
        mutation_rate: float | None = None,
        elitism: int = 0,
        selection_type: str = "tournament",
        tournament_size: int = 4,
        crossover_function: CrossoverFunction | None = None,
        mutation_function: MutationFunction | None = None,
        evaluation_backend: EvaluationBackend = "sequential",
        n_jobs: int | None = None,
        chunksize: int | None = None,
        triangle_alpha_range: population.AlphaRange = (
            population.TRIANGLE_ALPHA_RANGE
        ),
        n_triangles: int = population.N_TRIANGLES,
        initial_population: list[Individual] | None = None,
        seeded: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Configures a genetic image approximation run.

        Args:
            target:               RGB target image with shape (height, width, 3).
            fitness_function:     Callable: (target_array, generated_array) → float.
            population_size:      Number of individuals evaluated each generation.
            generations:          Number of generations to run.
            crossover_rate:       Probability of crossover per pair (0–1).
                                  Required when crossover_function is provided.
            mutation_rate:        Per-triangle mutation probability (0–1).
                                  Required when mutation_function is provided.
            elitism:              Number of best individuals copied unchanged to
                                  the next generation (0 = no elitism).
            selection_type:       Parent selection strategy: "tournament",
                                  "ranking", or "roulette".
            tournament_size:      Number of individuals sampled per tournament
                                  (only used when selection_type="tournament").
            crossover_function:   Operator: (p1, p2, rate) → child or children.
            mutation_function:    Operator: (individual, rate, w, h, alpha) → individual.
            evaluation_backend:   "sequential", "thread", or "process".
            n_jobs:               Worker count for thread/process backends.
            chunksize:            Batch size for the process pool.
            triangle_alpha_range: Inclusive (min, max) alpha values for triangles.
            n_triangles:          Triangles per individual (default 100).
            initial_population:   Optional explicit generation-0 population.
                                  Useful for warm-starting from a prior run.
            seeded:               If True, initialise triangle colours by sampling
                                  random pixels from the target image.
            progress_callback:    Optional callable receiving each GenerationLog.

        Raises:
            ValueError: If any parameter fails validation.
        """

        # --- Shape / range validation ---
        if target.ndim != 3 or target.shape[2] != 3:
            raise ValueError("target must have shape (H, W, 3).")
        if population_size <= 0:
            raise ValueError("population_size must be greater than zero.")
        if generations <= 0:
            raise ValueError("generations must be greater than zero.")
        if not 0 <= elitism <= population_size:
            raise ValueError("elitism must be between 0 and population_size.")
        if n_triangles <= 0:
            raise ValueError("n_triangles must be greater than zero.")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable when provided.")

        # --- Backend validation ---
        normalized_backend = evaluation.normalize_evaluation_backend(
            evaluation_backend
        )
        evaluation.validate_optional_positive_int(n_jobs, "n_jobs")
        evaluation.validate_optional_positive_int(chunksize, "chunksize")

        # Process backend requires picklable fitness functions because jobs
        # are serialised and sent across process boundaries
        if normalized_backend == "process":
            evaluation.validate_process_fitness_function(fitness_function)

        self.target = target.astype(np.float32)
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.generations = generations
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function

        self.crossover_rate = self._resolve_operator_rate(
            "crossover_rate", crossover_rate, self.crossover_function
        )
        self.mutation_rate = self._resolve_operator_rate(
            "mutation_rate", mutation_rate, self.mutation_function
        )

        self.elitism = elitism
        self.selection_type = selection.normalize_selection_type(selection_type)
        self.tournament_size = tournament_size
        self.evaluation_backend = normalized_backend
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.triangle_alpha_range = population.validate_triangle_alpha_range(
            triangle_alpha_range
        )
        self.n_triangles = n_triangles
        self.initial_population = self._normalize_initial_population(
            initial_population,
            n_triangles=n_triangles,
        )
        self.seeded = seeded
        self.progress_callback = progress_callback

        self.image_height, self.image_width = self.target.shape[:2]

        # Mutable run state — reset by initialize()
        self.population: list[Individual] = []
        self.best_individual: Individual | None = None
        self.best_fitness = float("inf")
        self.history: list[float] = []

    # -----------------------------------------------------------------------
    # Public lifecycle methods
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Creates the generation-0 population and resets all run state.

        If an ``initial_population`` was supplied at construction time, it is
        used directly (padded with random individuals if it is smaller than
        ``population_size``).  Otherwise a fresh random population is created.

        Calling ``initialize()`` again resets the run completely — useful for
        running the GA multiple times with the same configuration.
        """

        if self.initial_population is None:
            # No warm-start — generate a fully random population
            self.population = population.create_population(
                population_size=self.population_size,
                n_triangles=self.n_triangles,
                image_width=self.image_width,
                image_height=self.image_height,
                triangle_alpha_range=self.triangle_alpha_range,
                target=self.target,
                seeded=self.seeded,
            )
        else:
            # Use the provided population as the starting point
            seeded = copy.deepcopy(self.initial_population[: self.population_size])

            # If the provided population is smaller, pad it with random individuals
            if len(seeded) < self.population_size:
                seeded.extend(
                    population.create_population(
                        population_size=self.population_size - len(seeded),
                        n_triangles=self.n_triangles,
                        image_width=self.image_width,
                        image_height=self.image_height,
                        triangle_alpha_range=self.triangle_alpha_range,
                        target=self.target,
                        seeded=self.seeded,
                    )
                )
            self.population = seeded

        self.best_individual = None
        self.best_fitness = float("inf")
        self.history = []

    def evaluate(self) -> list[float]:
        """
        Computes fitness for all individuals in the current population.

        Also updates ``best_individual`` and ``best_fitness`` if any individual
        in the current population beats the previously tracked best.

        Returns:
            List of fitness floats in population order (lower = better).
        """
        return self._evaluate_population()

    # -----------------------------------------------------------------------
    # Private helpers used inside run()
    # -----------------------------------------------------------------------

    def _evaluate_population(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """Evaluate the population and update the tracked global best."""

        if not self.population:
            raise ValueError(
                "Population is empty. Call initialize() before evaluate()."
            )

        fitness_values = self._compute_population_fitness(executor)

        for individual, fitness_value in zip(
            self.population, fitness_values, strict=True
        ):
            if fitness_value < self.best_fitness:
                self.best_fitness    = fitness_value
                self.best_individual = copy.deepcopy(individual)

        return fitness_values

    def _compute_population_fitness(
        self,
        executor: evaluation.Executor | None = None,
    ) -> list[float]:
        """Dispatch fitness computation to the configured backend."""

        if self.evaluation_backend == "sequential":
            return evaluation.compute_population_fitness_sequential(
                self.population,
                self.target,
                self.fitness_function,
                self.image_width,
                self.image_height,
            )

        if executor is None:
            with evaluation.create_evaluation_executor(
                self.evaluation_backend,
                self.n_jobs,
                self.target,
                self.fitness_function,
                self.image_width,
                self.image_height,
            ) as managed_executor:
                return evaluation.compute_population_fitness_with_executor(
                    managed_executor,
                    self.evaluation_backend,
                    self.population,
                    self.target,
                    self.fitness_function,
                    self.image_width,
                    self.image_height,
                    self.chunksize,
                )

        return evaluation.compute_population_fitness_with_executor(
            executor,
            self.evaluation_backend,
            self.population,
            self.target,
            self.fitness_function,
            self.image_width,
            self.image_height,
            self.chunksize,
        )

    def select_parents(
        self,
        fitness_values: list[float],
    ) -> tuple[Individual, Individual]:
        """
        Select two parents independently. Subclasses override this for restricted mating.
        """

        return self.select_parent(fitness_values), self.select_parent(fitness_values)

    def select_parent(self, fitness_values: list[float]) -> Individual:
        """Select one parent using the configured selection strategy."""

        return selection.select_parent(
            self.population,
            fitness_values,
            selection_type=self.selection_type,
            tournament_size=self.tournament_size,
        )

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> list[Individual]:
        """
        Apply crossover and return a list of children (deep-copied).

        Normalises operator output (single child, tuple, or list) to a flat list.
        If no crossover function is configured, one parent is cloned at random.
        """

        if self.crossover_function is None:
            fallback_parent = parent1 if np.random.random() < 0.5 else parent2
            return [copy.deepcopy(fallback_parent)]

        crossover_result = self.crossover_function(
            parent1,
            parent2,
            self.crossover_rate,
        )

        if isinstance(crossover_result, tuple):
            children = list(crossover_result)
        elif crossover_result and isinstance(crossover_result[0], population.Triangle):
            children = [crossover_result]
        else:
            children = list(crossover_result)

        if not children:
            raise ValueError("crossover_function must return at least one child.")

        return copy.deepcopy(children)

    def mutate(self, individual: Individual) -> tuple[Individual, int]:
        """Apply mutation and return (mutated_individual, changed_triangle_count)."""

        if self.mutation_function is None:
            return individual, 0

        # Snapshot before mutation to count how many triangles changed
        before_mutation = copy.deepcopy(individual)
        mutated_individual = self.mutation_function(
            individual,
            self.mutation_rate,
            self.image_width,
            self.image_height,
            self.triangle_alpha_range,
        )
        changed_triangles = self._count_changed_triangles(
            before_mutation=before_mutation,
            after_mutation=mutated_individual,
        )

        return mutated_individual, changed_triangles

    def _emit_progress(self, generation_log: logs.GenerationLog) -> None:
        if self.progress_callback is not None:
            self.progress_callback(generation_log)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        patience: int | None = None,
        min_delta: float = 0.0,
    ) -> tuple[float, list[float]]:
        """
        Runs the GA and returns the best fitness and convergence history.

        Args:
            patience:  Stop early if the global best does not improve by more
                       than min_delta for this many consecutive generations.
                       None (default) disables early stopping.
            min_delta: Minimum absolute improvement in RMSE that counts as
                       progress. Only used when patience is set.

        After run() returns:
          - ``ga.best_individual`` holds the best triangle list found.
          - ``ga.history`` holds the global best RMSE after each generation.
        """

        self.initialize()

        executor: evaluation.Executor | None = None
        stale = 0
        best_at_last_check = float(self.best_fitness)

        try:
            if self.evaluation_backend != "sequential":
                executor = evaluation.create_evaluation_executor(
                    self.evaluation_backend,
                    self.n_jobs,
                    self.target,
                    self.fitness_function,
                    self.image_width,
                    self.image_height,
                )

            for generation in range(self.generations):
                evaluation_started = time.perf_counter()
                fitness_values = self._evaluate_population(executor)
                evaluation_duration_seconds = (
                    time.perf_counter() - evaluation_started
                )

                ranked_indices          = np.argsort(fitness_values)
                generation_best_fitness = float(fitness_values[int(ranked_indices[0])])
                global_best_fitness     = float(self.best_fitness)

                self.history.append(global_best_fitness)

                if patience is not None:
                    if best_at_last_check - global_best_fitness > min_delta:
                        best_at_last_check = global_best_fitness
                        stale = 0
                    else:
                        stale += 1
                    if stale >= patience:
                        break

                is_last = generation == self.generations - 1
                offspring_created   = 0
                mutated_offspring   = 0
                mutated_triangles   = 0
                if not is_last:
                    next_population = [
                        copy.deepcopy(self.population[int(index)])
                        for index in ranked_indices[: self.elitism]
                    ]

                    while len(next_population) < self.population_size:
                        parent1, parent2 = self.select_parents(fitness_values)
                        children = self.crossover(parent1, parent2)

                        for child in children:
                            if len(next_population) >= self.population_size:
                                break
                            child, changed = self.mutate(child)
                            next_population.append(child)
                            offspring_created += 1
                            if changed > 0:
                                mutated_offspring += 1
                                mutated_triangles += changed

                    self.population = next_population[: self.population_size]

                if self.progress_callback is not None:
                    self._emit_progress(logs.create_generation_log(
                        generation=generation,
                        generation_best_fitness=generation_best_fitness,
                        generation_mean_fitness=float(np.mean(fitness_values)),
                        global_best_fitness=global_best_fitness,
                        evaluation_backend=self.evaluation_backend,
                        n_jobs=self.n_jobs,
                        chunksize=self.chunksize,
                        evaluation_duration_seconds=evaluation_duration_seconds,
                        mutation_rate_used=float(self.mutation_rate),
                        offspring_created=offspring_created,
                        mutated_offspring=mutated_offspring,
                        mutated_triangles=mutated_triangles,
                    ))

        finally:
            if executor is not None:
                executor.shutdown()

        if self.best_individual is None:
            raise RuntimeError(
                "The genetic algorithm did not produce a best individual."
            )

        return float(self.best_fitness), list(self.history)

    # -----------------------------------------------------------------------
    # Introspection helpers
    # -----------------------------------------------------------------------

    def params_dict(self) -> dict:
        """Return a JSON-serialisable dict of all configuration parameters."""
        return {
            "population_size":      self.population_size,
            "generations":          self.generations,
            "n_triangles":          self.n_triangles,
            "crossover_rate":       self.crossover_rate,
            "mutation_rate":        self.mutation_rate,
            "mutation_function":    getattr(self.mutation_function, "__name__", None),
            "crossover_function":   getattr(self.crossover_function, "__name__", None),
            "elitism":              self.elitism,
            "selection_type":       self.selection_type,
            "tournament_size":      self.tournament_size,
            "triangle_alpha_range": list(self.triangle_alpha_range),
            "evaluation_backend":   self.evaluation_backend,
            "n_jobs":               self.n_jobs,
            "initialization":       "seeded" if self.seeded else "random",
        }
