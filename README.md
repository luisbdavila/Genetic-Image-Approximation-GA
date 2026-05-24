# CIFO Project — Triangle Image Approximation with Genetic Algorithms

Approximating Vermeer's *Girl with a Pearl Earring* using a population of 100 opaque triangles, evolved by a genetic algorithm to minimise the pixel-level RMSE against the 300×400 target image.

## Setup

```bash
uv sync
```

Then open any notebook with the project interpreter. The first cell adds the project root to `sys.path` automatically.

## Notebooks

| Notebook | Purpose |
|---|---|
| `Initial_analysis.ipynb` | First look at the target image,|
| `Grid_search_experiment.ipynb` | Cartesian-product search over crossover, selection, and restricted mating combinations |

## Project Structure

```
src/
  load_image.py        — loads and resizes the target image to a NumPy array
  population.py        — Triangle dataclass, population factories, alpha sampling
  rendering.py         — renders a triangle list to a PIL image / NumPy array
  statistical_analysis.py — Statistical functions for hypotesis tests
  ga/
    algorithm.py       — GeneticAlgorithm: selection → crossover → mutation → elitism loop
    fitness.py         — compute_rmse: pixel-level RMSE fitness function
    cross_over.py      — six crossover operators (single-point, two-point, PMX, cycle; 1- and 2-child variants)
    mutate.py          — random_triangle_mutation operator
    selection.py       — tournament, ranking, and roulette-wheel parent selection
    diversity.py       — FitnessSharingGA, RestrictedMatingGA, FitnessSharingRestrictedMatingGA; diversity trial runners
    parallel.py        — GAConfig, run_trials, run_grid_search, run_variants_batch (ProcessPoolExecutor + JSON caching)
    plotting.py        — reusable matplotlib figures: convergence curves, bar charts, diversity panels
    evaluation.py      — sequential, thread, and process fitness evaluation backends
    logs.py            — GenerationLog type and builder (used by progress_callback)
notebooks/
images/
  girl_pearl_earing.png   — 300×400 target image
results/                  — cached JSON trial results, organised by experiment
```

## GA Components

**Representation** — each individual is a list of 100 triangles. Every triangle has 9 parameters: three vertex coordinates (x, y) and an RGB colour. Alpha is fixed at 255 (fully opaque) in all experiments.

**Fitness** — pixel-by-pixel RMSE between the rendered candidate and the target, normalised to [0, 1]. Lower is better.

**Crossover operators** — `single_point`, `single_point_two_children`, `two_point`, `two_point_two_children`, `pmx`, `cycle`.

**Mutation** — `random_triangle_mutation`: replaces each triangle's geometry or colour with a random value at a per-triangle probability.

**Selection** — tournament (k=3), ranking, roulette-wheel.

**Diversity methods** — fitness sharing (shared fitness via niche radius σ) and restricted mating (distance-based parent pairing). A combined variant (`FitnessSharingRestrictedMatingGA`) is evaluated in the long-run section.

**Early stopping** — `ga.run(patience=N, min_delta=δ)` stops if the global best does not improve by more than δ for N consecutive generations.

## Caching

Every trial result is saved as a timestamped JSON file under `results/<experiment>/`. On re-run, the framework matches stored parameters against the current config (including crossover function name, mutation rate, population size, etc.) and skips any trial that already has a valid cache entry. Seeds are deterministic and derived from the full parameter dict, so re-running a cached config always produces the same result.

## Running an Experiment

```python
from src.ga.parallel import GAConfig, run_trials, run_grid_search
from src.ga import fitness, mutate, cross_over

config = GAConfig(
    target=target_array,
    fitness_function=fitness.compute_rmse,
    population_size=250,
    generations=300,
    crossover_function=cross_over.two_point_crossover_two_children,
    crossover_rate=0.9,
    mutation_function=mutate.random_triangle_mutation,
    mutation_rate=0.1,
    elitism=5,
    selection_type="tournament",
    triangle_alpha_range=(255, 255),
)

# 5 independent trials, results cached to disk
summary = run_trials(config, n_trials=5, pipeline="MyRun", results_dir=results_dir)
print(summary.mean_fitness, summary.std_fitness)

# Grid search over one parameter
results = run_grid_search(config, grid={"mutation_rate": [0.05, 0.1, 0.2]},
                          n_trials=5, pipeline_prefix="MutRate", results_dir=results_dir)
```
