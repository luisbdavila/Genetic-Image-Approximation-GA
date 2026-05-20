"""
Parallel GA trial runner — the main entry point for notebook experiments.

Why this module exists
----------------------
Running a single GA trial is straightforward, but for reliable results we
need to repeat each experiment multiple times (because GAs are stochastic)
and compare results across different parameter values (grid search).
Doing that naively in a notebook produces a lot of boilerplate and is slow.

This module solves both problems:
  1. ``run_trials``       — runs N independent trials of one config in parallel
                            and returns aggregated mean/std statistics.
  2. ``run_grid_search``  — runs N trials for each value in a parameter grid,
                            submitting all jobs to one pool so every CPU core
                            stays busy for the full duration.

Both functions check a file-based cache before running: if matching results
already exist on disk from a previous run, they are loaded instantly instead
of re-computing, saving significant time when iterating on analysis cells.

Parallelism design
------------------
macOS uses the "spawn" start method for multiprocessing, which means each
worker process is a fresh Python interpreter.  Functions passed to workers
must be picklable — module-level functions are always picklable, but lambdas
and functions defined in notebook cells are not.

To avoid this:
  - ``GAConfig`` is a dataclass with only serialisable fields.
  - ``run_single_ga`` is a module-level function.
  - Inside each worker, the GA uses ``evaluation_backend="sequential"``
    to prevent the worker from spawning its own child processes (nested pools
    cause deadlocks or resource exhaustion on macOS).

The outer ``ProcessPoolExecutor`` in ``run_trials`` / ``run_grid_search``
provides the parallelism, with one process per trial.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import population

Individual = list[population.Triangle]


# ---------------------------------------------------------------------------
# Serialisation helpers (previously in results.py)
# ---------------------------------------------------------------------------

def _individual_to_json(individual: Individual) -> list[dict]:
    return [
        {"x1": t.x1, "y1": t.y1, "x2": t.x2, "y2": t.y2,
         "x3": t.x3, "y3": t.y3, "r": t.r, "g": t.g, "b": t.b, "a": t.a}
        for t in individual
    ]


def _individual_from_json(data: list[dict]) -> Individual:
    return [
        population.Triangle(
            x1=t["x1"], y1=t["y1"], x2=t["x2"], y2=t["y2"],
            x3=t["x3"], y3=t["y3"], r=t["r"], g=t["g"], b=t["b"], a=t["a"],
        )
        for t in data
    ]


def save_run(
    pipeline: str,
    parameters: dict,
    best_fitness: float,
    history: list[float],
    best_individual: Individual,
    runtime_seconds: float,
    notes: str = "",
    results_dir: Path | str = Path("results"),
) -> Path:
    """
    Saves one GA trial result to a timestamped JSON file.

    Args:
        pipeline:         Experiment label used as a filename prefix and cache key.
        parameters:       Dict of GA parameters to store alongside the result.
        best_fitness:     Final RMSE of the best individual found.
        history:          Per-generation best-RMSE list.
        best_individual:  List of Triangle objects that achieved best_fitness.
        runtime_seconds:  Wall-clock seconds the trial took.
        notes:            Optional free-text annotation (e.g. trial index).
        results_dir:      Directory to write the JSON file into (created if needed).

    Returns:
        Path to the written JSON file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    run_id = f"{pipeline.lower().replace(' ', '_')}_{timestamp}"
    filename = results_dir / f"{run_id}.json"
    tmp_filename = filename.with_suffix(".json.tmp")
    payload = {
        "run_id": run_id,
        "pipeline": pipeline,
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": round(runtime_seconds, 1),
        "notes": notes,
        "parameters": parameters,
        "results": {
            "best_fitness": best_fitness,
            "generations_run": len(history),
            "history": history,
        },
        "best_individual": _individual_to_json(best_individual),
    }
    # Write to a temp file first, then atomically rename — prevents a corrupt
    # .json file if the process is killed mid-write
    tmp_filename.write_text(json.dumps(payload, indent=2))
    tmp_filename.rename(filename)
    return filename


def load_all_runs(results_dir: Path | str) -> list[dict]:
    """
    Loads all saved trial JSONs from a directory and returns them as a list.

    Each returned dict includes all fields from the saved JSON plus an
    ``individual`` key that is already deserialized back to Triangle objects.
    Files that cannot be parsed are silently skipped.

    Args:
        results_dir: Directory to scan for ``*.json`` files.

    Returns:
        List of run dicts sorted by filename (oldest first).
        Returns an empty list if the directory does not exist.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["individual"] = _individual_from_json(data["best_individual"])
            runs.append(data)
        except Exception:
            pass
    return runs


# ---------------------------------------------------------------------------
# GAConfig — fully self-contained configuration for one GA trial
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """
    All parameters needed to run one GA trial, stored in a serialisable dataclass.

    Using a dataclass instead of keyword arguments makes it easy to override
    a single parameter for a grid search::

        config = GAConfig(**{**base_config.__dict__, "elitism": 3})

    All fields except ``trial`` and ``label`` directly correspond to
    GeneticAlgorithm constructor parameters.

    Attributes:
        target:               RGB target image array (H, W, 3).
        fitness_function:     Module-level fitness callable (must be picklable).
        population_size:      Number of individuals per generation.
        generations:          Number of generations to evolve.
        crossover_function:   Module-level crossover callable.
        crossover_rate:       Crossover probability per parent pair (0–1).
        mutation_function:    Module-level mutation callable.
        mutation_rate:        Per-triangle mutation probability (0–1).
        elitism:              Best N individuals copied unchanged each generation.
        selection_type:       "tournament", "ranking", or "roulette".
        tournament_size:      Candidates sampled per tournament (only used when
                              selection_type="tournament").
        triangle_alpha_range: Inclusive (min, max) alpha range for triangles.
        trial:                0-based trial index (set automatically by run_trials).
        label:                Pipeline name injected into log messages (set automatically).
    """

    target: np.ndarray
    fitness_function: Any
    population_size: int
    generations: int
    crossover_function: Any
    crossover_rate: float
    mutation_function: Any
    mutation_rate: float
    elitism: int
    selection_type: str
    tournament_size: int = 4
    triangle_alpha_range: tuple[int, int] = (255, 255)
    trial: int = 0    # set automatically — do not set manually
    label: str = ""   # set automatically — do not set manually


# ---------------------------------------------------------------------------
# TrialSummary — aggregated statistics returned to notebook cells
# ---------------------------------------------------------------------------

@dataclass
class TrialSummary:
    """
    Aggregated statistics from N independent GA trials of the same config.

    Attributes:
        pipeline:        Pipeline label (e.g. "Baseline-GA", "Elitism-3").
        n_trials:        Number of trials that were averaged.
        mean_fitness:    Mean final RMSE across all trials.
        std_fitness:     Standard deviation of final RMSE across trials.
        min_fitness:     Best (lowest) RMSE seen across all trials.
        max_fitness:     Worst (highest) RMSE seen across all trials.
        mean_history:    Per-generation mean RMSE averaged across trials.
        std_history:     Per-generation std of RMSE across trials.
        best_individual: The triangle individual that achieved min_fitness.
        all_fitness:     Raw list of final RMSE values, one per trial.
        total_runtime:   Total wall-clock time for all trials combined.
    """

    pipeline: str
    n_trials: int
    mean_fitness: float
    std_fitness: float
    min_fitness: float
    max_fitness: float
    mean_history: list[float]
    std_history: np.ndarray
    best_individual: Individual
    all_fitness: list[float]
    total_runtime: float

    def __str__(self) -> str:
        return (
            f"{self.pipeline} — {self.n_trials} trials\n"
            f"  Mean RMSE : {self.mean_fitness:.6f} ± {self.std_fitness:.6f}\n"
            f"  Best      : {self.min_fitness:.6f}  |  Worst: {self.max_fitness:.6f}\n"
            f"  Runtime   : {self.total_runtime:.0f}s total"
        )


# ---------------------------------------------------------------------------
# Cache helpers — check disk before running expensive trials
# ---------------------------------------------------------------------------

def _params_match(config: GAConfig, run: dict) -> bool:
    """
    Returns True if a saved JSON run's parameters match the current GAConfig.

    Compares the key numeric and string parameters that define the experiment
    setup.  If all match, the saved run is considered a valid cache hit for
    this config.

    The fields compared are: population_size, generations, crossover_rate,
    mutation_rate, elitism, selection_type, triangle_alpha_range,
    crossover_function name, and mutation_function name.

    Args:
        config: The current GAConfig to check against.
        run:    A loaded run dict as returned by load_all_runs().

    Returns:
        True if all key parameters match, False otherwise.
    """

    p = run.get("parameters", {})
    return (
        p.get("population_size")    == config.population_size
        and p.get("generations")    == config.generations
        and p.get("crossover_rate") == config.crossover_rate
        and p.get("mutation_rate")  == config.mutation_rate
        and p.get("elitism")        == config.elitism
        and p.get("selection_type") == config.selection_type
        # tournament_size is optional: old cached runs may not have it serialised
        and (p.get("tournament_size") is None or p.get("tournament_size") == config.tournament_size)
        # alpha range is stored as a list in JSON, but as a tuple in GAConfig
        and p.get("triangle_alpha_range") == list(config.triangle_alpha_range)
        # compare by function name so the check works across Python sessions
        and p.get("crossover_function") == getattr(config.crossover_function, "__name__", "")
        and p.get("mutation_function")  == getattr(config.mutation_function,  "__name__", "")
    )


def _get_ga_class(name: str):
    """Import and return a GA class by name — called inside worker processes."""
    if name == "GeneticAlgorithm":
        from .algorithm import GeneticAlgorithm
        return GeneticAlgorithm
    if name == "FitnessSharingGA":
        from .diversity import FitnessSharingGA
        return FitnessSharingGA
    if name == "RestrictedMatingGA":
        from .diversity import RestrictedMatingGA
        return RestrictedMatingGA
    if name == "FitnessSharingRestrictedMatingGA":
        from .diversity import FitnessSharingRestrictedMatingGA
        return FitnessSharingRestrictedMatingGA
    raise ValueError(f"Unknown GA class: {name!r}")


def _params_match_variant(config: GAConfig, run: dict, extra_kwargs: dict) -> bool:
    """True if run matches config base params AND the extra_kwargs values."""
    if not _params_match(config, run):
        return False
    p = run.get("parameters", {})
    skip = {"target", "fitness_function", "crossover_function", "mutation_function"}
    return all(p.get(k) == v for k, v in extra_kwargs.items() if k not in skip)


def _load_cached_results_variant(
    pipeline: str,
    results_dir: Path | str,
    config: GAConfig,
    extra_kwargs: dict,
) -> list[dict]:
    """Like _load_cached_results but also checks extra_kwargs values."""

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    loaded = []
    for r in load_all_runs(results_dir):
        if r.get("pipeline") == pipeline and _params_match_variant(config, r, extra_kwargs):
            loaded.append({
                "fitness":    r["results"]["best_fitness"],
                "history":    r["results"]["history"],
                "individual": r["individual"],
                "runtime":    r.get("runtime_seconds", 0),
                "trial":      len(loaded),
                "label":      pipeline,
                "params":     r.get("parameters", {}),
            })
    return loaded


def _load_cached_results(
    pipeline: str,
    results_dir: Path | str,
    config: GAConfig,
) -> list[dict]:
    """
    Returns all saved runs that match the pipeline name and config parameters.

    Each returned dict uses the same keys as ``run_single_ga`` output so that
    cached and freshly-computed results can be combined uniformly:
    ``fitness``, ``history``, ``individual``, ``runtime``, ``trial``,
    ``label``, ``params``.

    Returns an empty list when the directory does not exist or no runs match.

    Args:
        pipeline:    Pipeline name to search for (e.g. "Baseline-GA").
        results_dir: Directory containing saved JSON files.
        config:      Current GAConfig to match against saved parameters.

    Returns:
        List of normalised run dicts, sorted oldest-first.
    """


    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    loaded = []
    for r in load_all_runs(results_dir):
        if r.get("pipeline") == pipeline and _params_match(config, r):
            loaded.append({
                "fitness":    r["results"]["best_fitness"],
                "history":    r["results"]["history"],
                "individual": r["individual"],
                "runtime":    r.get("runtime_seconds", 0),
                "trial":      len(loaded),
                "label":      pipeline,
                "params":     r.get("parameters", {}),
            })
    return loaded


def _build_trial_summary(
    pipeline: str,
    results: list[dict],
    total_runtime: float,
) -> TrialSummary:
    """
    Builds a TrialSummary from a list of run dicts (cached or fresh).

    Args:
        pipeline:      Pipeline label for the summary.
        results:       List of dicts with keys fitness, history, individual.
        total_runtime: Combined wall-clock seconds for all runs.

    Returns:
        Aggregated TrialSummary across all supplied results.
    """

    fitnesses   = [r["fitness"]    for r in results]
    histories   = [r["history"]    for r in results]
    individuals = [r["individual"] for r in results]

    return TrialSummary(
        pipeline=pipeline,
        n_trials=len(results),
        mean_fitness=float(np.mean(fitnesses)),
        std_fitness=float(np.std(fitnesses)),
        min_fitness=float(np.min(fitnesses)),
        max_fitness=float(np.max(fitnesses)),
        mean_history=np.mean(histories, axis=0).tolist(),
        std_history=np.std(histories, axis=0),
        best_individual=individuals[int(np.argmin(fitnesses))],
        all_fitness=fitnesses,
        total_runtime=total_runtime,
    )


# ---------------------------------------------------------------------------
# Worker function — must be at module level to be picklable by spawn workers
# ---------------------------------------------------------------------------

def run_single_ga(config: GAConfig) -> dict:
    """
    Runs one complete GA trial and returns results as a plain serialisable dict.

    This function is the unit of work submitted to ``ProcessPoolExecutor``.
    It must be defined at module level (not inside a class or another function)
    so that Python's multiprocessing can pickle and send it to worker processes
    — this is a requirement of the macOS "spawn" start method.

    The GA inside this worker uses ``evaluation_backend="sequential"`` to avoid
    nested process pools: the outer pool (in run_trials / run_grid_search) already
    provides CPU parallelism across trials, so each individual trial evaluates
    its population sequentially on its assigned core.

    Args:
        config: Fully populated GAConfig for this specific trial.

    Returns:
        Dict with keys: label, trial, elitism, fitness, history,
        individual, runtime, params.
    """

    # Deferred import inside the worker to avoid import-time side effects
    # when the module is first loaded in the main process
    from .algorithm import GeneticAlgorithm

    print(f"  → [{config.label}] Trial {config.trial + 1} starting  "
          f"(pop={config.population_size}, gen={config.generations})")

    ga = GeneticAlgorithm(
        target=config.target,
        fitness_function=config.fitness_function,
        population_size=config.population_size,
        generations=config.generations,
        crossover_function=config.crossover_function,
        crossover_rate=config.crossover_rate,
        mutation_function=config.mutation_function,
        mutation_rate=config.mutation_rate,
        elitism=config.elitism,
        selection_type=config.selection_type,
        tournament_size=config.tournament_size,
        triangle_alpha_range=config.triangle_alpha_range,
        evaluation_backend="sequential",
    )

    t0 = time.perf_counter()
    best_fitness, history = ga.run()
    runtime = time.perf_counter() - t0

    print(f"  ✓ [{config.label}] Trial {config.trial + 1} done     "
          f"RMSE={best_fitness:.6f}  ({runtime:.1f}s)")

    return {
        "label":      config.label,
        "trial":      config.trial,
        "elitism":    config.elitism,
        "fitness":    best_fitness,
        "history":    history,
        "individual": ga.best_individual,
        "runtime":    runtime,
        "params":     ga.params_dict(),
    }


def run_single_ga_variant(config: GAConfig, ga_class_name: str, extra_kwargs: dict) -> dict:
    """
    Worker that runs one trial of any GA subclass.  Must be module-level for pickling.

    Like ``run_single_ga`` but instantiates the class named by ``ga_class_name``
    and passes ``extra_kwargs`` (e.g. ``sigma_share``, ``mating_type``) to it.

    Args:
        config:        Fully populated GAConfig for this specific trial.
        ga_class_name: One of ``"GeneticAlgorithm"``, ``"FitnessSharingGA"``,
                       ``"RestrictedMatingGA"``, ``"FitnessSharingRestrictedMatingGA"``.
        extra_kwargs:  Subclass-specific keyword arguments (e.g.
                       ``{"sigma_share": 0.3, "n_bins": 8}``).

    Returns:
        Same dict shape as ``run_single_ga``.
    """
    cls = _get_ga_class(ga_class_name)
    print(f"  → [{config.label}] Trial {config.trial + 1} starting  "
          f"(pop={config.population_size}, gen={config.generations})")

    ga = cls(
        target=config.target,
        fitness_function=config.fitness_function,
        population_size=config.population_size,
        generations=config.generations,
        crossover_function=config.crossover_function,
        crossover_rate=config.crossover_rate,
        mutation_function=config.mutation_function,
        mutation_rate=config.mutation_rate,
        elitism=config.elitism,
        selection_type=config.selection_type,
        tournament_size=config.tournament_size,
        triangle_alpha_range=config.triangle_alpha_range,
        evaluation_backend="sequential",
        **extra_kwargs,
    )

    t0 = time.perf_counter()
    best_fitness, history = ga.run()
    runtime = time.perf_counter() - t0

    print(f"  ✓ [{config.label}] Trial {config.trial + 1} done     "
          f"RMSE={best_fitness:.6f}  ({runtime:.1f}s)")

    skip = {"target", "fitness_function", "crossover_function", "mutation_function"}
    return {
        "label":      config.label,
        "trial":      config.trial,
        "fitness":    best_fitness,
        "history":    history,
        "individual": ga.best_individual,
        "runtime":    runtime,
        "params":     {
            **ga.params_dict(),
            **{k: v for k, v in extra_kwargs.items() if k not in skip},
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_trials(
    config: GAConfig,
    n_trials: int,
    pipeline: str,
    results_dir: Path | str,
    notes: str = "",
    max_workers: int | None = None,
) -> TrialSummary:
    """
    Runs N independent trials of one GA configuration and returns aggregated statistics.

    Trials run in parallel using a ProcessPoolExecutor.  Before running, the
    function checks the disk cache: only the trials that are missing are run;
    already-saved results are loaded and merged with the new ones.

    Args:
        config:      Base GA configuration.  ``trial`` and ``label`` are
                     set automatically for each worker.
        n_trials:    Total number of independent trials desired.
        pipeline:    Label written into each saved JSON and used as the
                     cache key (e.g. ``"Baseline-GA"``).
        results_dir: Directory where individual trial JSONs are saved.
        notes:       Free-text note appended to every saved run.
        max_workers: Number of parallel processes.  Defaults to the number
                     of missing trials (one process per missing trial).

    Returns:
        TrialSummary with mean/std RMSE, convergence history, and best individual.
    """

    results_dir = Path(results_dir)

    # Load however many matching trials are already on disk
    existing = _load_cached_results(pipeline, results_dir, config)
    n_existing = len(existing)

    if n_existing >= n_trials:
        # Full cache hit — take the most recent n_trials runs
        runs = existing[-n_trials:]
        summary = _build_trial_summary(pipeline, runs,
                                       sum(r["runtime"] for r in runs))
        print(f"✓ '{pipeline}' — loaded {n_trials} cached trials  "
              f"(mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f})")
        return summary

    # Partial or cold cache — only run the missing trials
    n_missing = n_trials - n_existing
    workers = max_workers or min(n_missing, os.cpu_count() or 1)

    if n_existing:
        print(f"~ '{pipeline}' — {n_existing} cached + {n_missing} missing  "
              f"pop={config.population_size}  gen={config.generations}  workers={workers}")
    else:
        print(f"┌─ '{pipeline}'  {n_trials} trials  "
              f"pop={config.population_size}  gen={config.generations}  "
              f"workers={workers}")

    # Trial indices continue from where the cached runs left off
    configs = [
        GAConfig(**{**config.__dict__, "trial": n_existing + i, "label": pipeline})
        for i in range(n_missing)
    ]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_ga, c): c for c in configs}
        new_results = []
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as exc:
                cfg = futures[future]
                print(f"  ✗ [{cfg.label}] Trial {cfg.trial + 1} failed: {exc}", flush=True)
                continue
            # Save immediately so partial progress survives interruptions
            save_run(
                pipeline=pipeline,
                parameters=r["params"],
                best_fitness=r["fitness"],
                history=r["history"],
                best_individual=r["individual"],
                runtime_seconds=r["runtime"],
                notes=f"{notes} [trial {r['trial'] + 1}/{n_trials}]",
                results_dir=results_dir,
            )
            new_results.append(r)
    new_runtime = time.perf_counter() - t0

    new_results.sort(key=lambda r: r["trial"])
    all_results = (existing + new_results)[-n_trials:]
    total_runtime = sum(r["runtime"] for r in existing) + new_runtime
    summary = _build_trial_summary(pipeline, all_results, total_runtime)

    print(f"└─ {summary.pipeline} — mean RMSE: {summary.mean_fitness:.6f} ± {summary.std_fitness:.6f}  "
          f"best: {summary.min_fitness:.6f}  ({new_runtime:.0f}s for new trials)")
    return summary



def run_variants_batch(
    config: GAConfig,
    variants: dict[str, tuple[str, str, dict]],
    n_trials: int,
    results_dir: Path | str,
    notes: str = "",
    max_workers: int | None = None,
) -> dict[str, TrialSummary]:
    """
    Run N trials for each variant, submitting all missing trials to a single
    shared ``ProcessPoolExecutor`` — the same pattern as ``run_grid_search``.

    All CPU cores stay busy for the full duration instead of being idle while
    the next variant's pool starts up.

    Args:
        config:      Base GA configuration shared by all variants.
        variants:    Ordered dict mapping display_label →
                     ``(pipeline, ga_class_name, extra_kwargs)``.
                     ``pipeline`` is the unique cache key written to disk.
        n_trials:    Target number of independent trials per variant.
        results_dir: Directory where individual trial JSONs are saved.
        notes:       Free-text note appended to every saved run.
        max_workers: Parallel processes.  Defaults to the total number of
                     missing trials across all variants.

    Returns:
        Dict mapping each display_label → ``TrialSummary``.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"┌─ {len(variants)} variants  {len(variants) * n_trials} trials total  "
          f"pop={config.population_size}  gen={config.generations}", flush=True)

    # ── Phase 1: check cache, collect all missing jobs ────────────────────────
    cached_by_label: dict[str, list[dict]] = {}
    all_jobs: list[tuple[str, str, str, dict, GAConfig]] = []
    # each job: (display_label, pipeline, ga_class_name, extra_kwargs, trial_config)

    for label, (pipeline, ga_class_name, extra_kwargs) in variants.items():
        existing   = _load_cached_results_variant(pipeline, results_dir, config, extra_kwargs)
        cached_by_label[label] = existing
        n_existing = len(existing)
        n_missing  = n_trials - n_existing

        if n_existing >= n_trials:
            s = _build_trial_summary(
                pipeline, existing[-n_trials:],
                sum(r["runtime"] for r in existing[-n_trials:]),
            )
            print(f"  ✓ '{pipeline}' — loaded {n_trials} cached  "
                  f"(mean RMSE={s.mean_fitness:.6f})", flush=True)
        else:
            tag = f"{n_existing} cached + {n_missing} missing" if n_existing else f"{n_missing} missing"
            print(f"  ~ '{pipeline}' — {tag}", flush=True)
            for i in range(n_missing):
                trial_cfg = GAConfig(
                    **{**config.__dict__, "trial": n_existing + i, "label": pipeline}
                )
                all_jobs.append((label, pipeline, ga_class_name, extra_kwargs, trial_cfg))

    # ── Full cache hit — no work to do ────────────────────────────────────────
    if not all_jobs:
        summaries = {}
        for label, (pipeline, _cls, _extra) in variants.items():
            runs = cached_by_label[label][-n_trials:]
            summaries[label] = _build_trial_summary(
                pipeline, runs, sum(r["runtime"] for r in runs)
            )
        print("└─ all cached", flush=True)
        return summaries

    # ── Phase 2: dispatch all missing trials to one pool ──────────────────────
    workers = max_workers or min(len(all_jobs), os.cpu_count() or 1)
    print(f"  dispatching {len(all_jobs)} new trials  workers={workers}", flush=True)

    t0 = time.perf_counter()
    new_by_label: dict[str, list[dict]] = {label: [] for label in variants}
    n_done = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {
            executor.submit(run_single_ga_variant, trial_cfg, ga_class_name, extra_kwargs):
                (label, pipeline, trial_cfg.trial)
            for label, pipeline, ga_class_name, extra_kwargs, trial_cfg in all_jobs
        }
        for future in as_completed(future_to_meta):
            label, pipeline, trial_idx = future_to_meta[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  ✗ [{pipeline}] Trial {trial_idx + 1} failed: {exc}", flush=True)
                continue
            n_done += 1
            print(f"  ✓ [{pipeline}] trial {trial_idx + 1}/{n_trials}  "
                  f"RMSE={result['fitness']:.6f}  ({result['runtime']:.1f}s)  "
                  f"[{n_done}/{len(all_jobs)} done]", flush=True)
            # Save immediately so partial progress survives interruptions
            save_run(
                pipeline=pipeline,
                parameters=result["params"],
                best_fitness=result["fitness"],
                history=result["history"],
                best_individual=result["individual"],
                runtime_seconds=result["runtime"],
                notes=f"{notes} [trial {trial_idx + 1}/{n_trials}]",
                results_dir=results_dir,
            )
            new_by_label[label].append(result)

    wall = time.perf_counter() - t0

    # ── Phase 3: build summaries (results already saved above) ───────────────
    summaries = {}
    for label, (pipeline, _cls, extra_kwargs) in variants.items():
        new_results = sorted(new_by_label[label], key=lambda r: r["trial"])
        all_results   = (cached_by_label[label] + new_results)[-n_trials:]
        total_runtime = (sum(r["runtime"] for r in cached_by_label[label])
                         + sum(r["runtime"] for r in new_results))
        summaries[label] = _build_trial_summary(pipeline, all_results, total_runtime)

    print(f"└─ done in {wall:.0f}s", flush=True)
    return summaries


def run_grid_search(
    base_config: GAConfig,
    grid: dict[str, list[Any]],
    n_trials: int,
    pipeline_prefix: str,
    results_dir: Path | str,
    notes: str = "",
    max_workers: int | None = None,
) -> dict[Any, TrialSummary]:
    """
    Runs a single-parameter grid search with all jobs dispatched to one pool.

    The key optimisation over a naive loop is that all ``(param_value × trial)``
    combinations are submitted to a *single* ``ProcessPoolExecutor`` rather than
    running one ``run_trials`` call per parameter value.  With 4 values × 5
    trials = 20 jobs and 12 CPU cores, all 20 jobs can overlap, using all cores
    for the full duration instead of wasting 7 cores while 5 run at a time.

    Per-value cache checking is still performed: any values that already have
    n_trials saved results are loaded from disk and excluded from the job list,
    so only the missing values are computed.

    Only one parameter key may be varied per call.

    Args:
        base_config:     Shared GA configuration.  The varied key is overridden
                         per job via ``**{**base_config.__dict__, param_name: value}``.
        grid:            Exactly one entry: ``{"param_name": [val1, val2, ...]}``.
        n_trials:        Number of independent trials per parameter value.
        pipeline_prefix: Prefix for pipeline labels.  E.g. ``"PopSize"`` produces
                         ``"PopSize-100"``, ``"PopSize-200"``, …
        results_dir:     Directory where individual trial JSONs are saved.
        notes:           Free-text note appended to every saved run.
        max_workers:     Parallel processes.  Defaults to ``os.cpu_count()``
                         (uses all logical CPU cores).

    Returns:
        Dict mapping each parameter value to its TrialSummary.

    Raises:
        ValueError: If grid does not contain exactly one key.
    """

    if len(grid) != 1:
        raise ValueError("run_grid_search expects exactly one key in `grid`.")

    param_name, param_values = next(iter(grid.items()))
    workers = max_workers or os.cpu_count() or 1
    results_dir = Path(results_dir)

    # --- Per-value cache check: separate fully-cached from needing work ---
    summaries: dict[Any, TrialSummary] = {}
    cached_per_value: dict[Any, list[dict]] = {}

    for value in param_values:
        pipeline = f"{pipeline_prefix}-{value}"
        cfg = GAConfig(**{**base_config.__dict__, param_name: value})
        existing = _load_cached_results(pipeline, results_dir, cfg)

        if len(existing) >= n_trials:
            runs = existing[-n_trials:]
            summary = _build_trial_summary(pipeline, runs,
                                           sum(r["runtime"] for r in runs))
            summaries[value] = summary
            print(f"✓ '{pipeline}' — loaded {n_trials} cached trials  "
                  f"(mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f})")
        else:
            cached_per_value[value] = existing

    if not cached_per_value:
        best_value = min(summaries, key=lambda v: summaries[v].mean_fitness)
        print(f"All {len(param_values)} values loaded from cache.  "
              f"Best {param_name}: {best_value}  "
              f"(mean RMSE={summaries[best_value].mean_fitness:.6f})")
        return summaries

    # --- Build flat job list: only the missing trials for each value ---
    all_configs: list[GAConfig] = []
    for value, existing in cached_per_value.items():
        pipeline = f"{pipeline_prefix}-{value}"
        n_missing = n_trials - len(existing)
        start_trial = len(existing)
        for i in range(n_missing):
            all_configs.append(
                GAConfig(**{**base_config.__dict__, param_name: value,
                            "trial": start_trial + i, "label": pipeline})
            )

    total_jobs = len(all_configs)
    print(f"┌─ Grid search  {param_name}={list(cached_per_value.keys())}")
    for value, existing in cached_per_value.items():
        n_missing = n_trials - len(existing)
        if existing:
            print(f"│  {pipeline_prefix}-{value}: {len(existing)} cached + {n_missing} new")
        else:
            print(f"│  {pipeline_prefix}-{value}: {n_trials} new trials")
    print(f"│  {total_jobs} jobs total  max_workers={workers}")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_ga, c): c for c in all_configs}
        all_new_unordered = []
        completed = 0
        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as exc:
                cfg = futures[future]
                print(f"│  ✗ [{cfg.label}] Trial {cfg.trial + 1} failed: {exc}", flush=True)
                continue
            # Save immediately so partial progress survives interruptions
            save_run(
                pipeline=r["label"],
                parameters=r["params"],
                best_fitness=r["fitness"],
                history=r["history"],
                best_individual=r["individual"],
                runtime_seconds=r["runtime"],
                notes=f"{notes} [trial {r['trial'] + 1}/{n_trials}]",
                results_dir=results_dir,
            )
            all_new_unordered.append(r)
            completed += 1
            print(f"│  {completed}/{total_jobs} jobs done")
    total_runtime = time.perf_counter() - t0

    print(f"│  All jobs done in {total_runtime:.0f}s")

    # Group new results by value label
    new_by_value: dict[Any, list[dict]] = {v: [] for v in cached_per_value}
    order = {(c.label, c.trial): i for i, c in enumerate(all_configs)}
    for r in sorted(all_new_unordered, key=lambda r: order[(r["label"], r["trial"])]):
        for value in cached_per_value:
            if r["label"] == f"{pipeline_prefix}-{value}":
                new_by_value[value].append(r)
                break

    # --- Build summaries by merging cached + new (results already saved above) ---
    for value in cached_per_value:
        pipeline = f"{pipeline_prefix}-{value}"
        existing    = cached_per_value[value]
        new_results = new_by_value[value]

        all_results = (existing + new_results)[-n_trials:]
        runtime = sum(r["runtime"] for r in existing) + (total_runtime / len(cached_per_value))
        summary = _build_trial_summary(pipeline, all_results, runtime)
        summaries[value] = summary
        print(f"│  {summary.pipeline:<20} mean RMSE={summary.mean_fitness:.6f} ± {summary.std_fitness:.6f}  "
              f"best={summary.min_fitness:.6f}")

    # Final summary line showing the winning parameter value
    best_value = min(param_values, key=lambda v: summaries[v].mean_fitness)
    print(f"└─ Best {param_name}: {best_value}  "
          f"(mean RMSE={summaries[best_value].mean_fitness:.6f})")
    return summaries
