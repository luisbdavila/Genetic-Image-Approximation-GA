# Step-by-Step Exploration

**Notebook:** [`notebooks/Step_by_step_exploration.ipynb`](../notebooks/Step_by_step_exploration.ipynb)

## Overview

This notebook implements **Additional Challenge 3** from the project requirements:

> *"For your GA implementation, consider some of its components (mutation, cross-over, selection) and evaluate their contribution to the overall result."*

Rather than tuning all Genetic Algorithm parameters simultaneously, the notebook follows a deliberate, incremental strategy: each algorithmic component is evaluated independently while the best settings found in all prior stages are carried forward. This cumulative approach isolates the contribution of each component and produces a well-justified final configuration.

## Structure

The exploration proceeds through the following sequential stages:

| Stage | What is evaluated |
|---|---|
| **Baseline** | Fixed starting configuration: population 300, 300 generations, single-point crossover, tournament selection, elitism, fully opaque triangles |
| **Population size** | Trade-off between exploration capacity and computational cost |
| **Elitism** | Balancing solution preservation against premature convergence |
| **Parent selection** | Tournament, ranking, and roulette strategies; followed by a tournament size sweep |
| **Crossover** | Operator type (single-point, two-point, cycle, PMX) and crossover rate |
| **Mutation rate** | Balance between preserving high-quality structures and introducing variation |
| **Diversity analysis** | Genotypic variance, genotypic entropy, phenotypic variance tracked across generations |
| **Fitness Sharing** | Niche-penalty diversity mechanism |
| **Restricted Mating** | Distance-based parent pairing (unidirectional, bidirectional, best partial match) |
| **Alpha range** | Triangle transparency — left last because it modifies rendering behaviour and increases search space complexity |
| **Cumulative summary** | Best configuration from each stage compared side by side |
| **Extended runs** | 3,000-generation experiments on the strongest configurations to assess long-horizon performance |

## Statistical Analysis

Every stage is evaluated across **at least 30 independent trials** to account for stochastic variation. A three-step statistical pipeline is applied to the final RMSE values at each stage:

1. **Shapiro-Wilk test** — checks normality of each configuration's results
2. **ANOVA / Kruskal-Wallis** — global test for any significant difference across configurations
3. **Welch t-test / Mann-Whitney U with Bonferroni correction** — pairwise comparisons to identify which specific configuration is better

If no stage-level statistical significance is found, the choice is guided by secondary factors such as computational speed or maintained diversity.

## Key Findings

- The most impactful single change was introducing **partially transparent triangles** (alpha range 5–128), which enabled smoother colour blending and reduced mean RMSE from 0.1419 to 0.1088.
- **Restricted Mating** (unidirectional) achieved the best result in the 3,000-generation extended run (RMSE 0.0856), outperforming standard GA, Fitness Sharing, and the combined approach.
- **Fitness Sharing** performed worse than standard GA in extended runs, suggesting the niche penalty was too strong for this image approximation task.
