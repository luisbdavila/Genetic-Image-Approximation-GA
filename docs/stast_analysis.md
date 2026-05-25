## Example: Complete Statistical Analysis Workflow

This example demonstrates how to use the statistical analysis module to:
1. Test normality of RMSE distributions
2. Apply appropriate statistical tests (parametric or non-parametric)
3. Compare final RMSE values
4. Determine if best configuration is significantly better than alternatives

### What the analysis does:

| Step | Test | Purpose |
|------|------|---------|
| 1 | **Shapiro-Wilk** | Determine if data is normally distributed |
| 2 | **Global Comparison** | ANOVA (if normal) or Kruskal-Wallis (if not) — are ANY differences significant? |
| 3 | **Pairwise Tests** | Welch's t-test or Mann-Whitney U with Bonferroni correction — which specific pairs differ? |

### Interpreting results:
- **p > 0.05**: No statistically significant difference (differences due to random variation)
- **p ≤ 0.05**: Statistically significant difference (reproducible/real difference)
- **Effect size**: Magnitude of the difference (important even if p is tiny)
- **Best config**: Lowest mean RMSE, but statistical significance confirms it's not a fluke

## Understanding the Statistical Analysis Results

### 1. **Normality Test (Shapiro-Wilk)**

**What it does:** Tests whether the distribution of final RMSE values (k trials per configuration) follows a normal (Gaussian) distribution.

**Hypothesis:**
- H₀ (null): Data is normally distributed
- H₁ (alternative): Data is NOT normally distributed

**Interpretation:**
- **p > 0.05** (✓ Normal): Use parametric tests (ANOVA, t-tests)
- **p ≤ 0.05** (✗ Not Normal): Use non-parametric tests (Kruskal-Wallis, Mann-Whitney U)

**Example:** If population sizes [100, 200, 250, 300] all have p > 0.05, then the RMSE values at each size come from a normal distribution, so ANOVA is appropriate.

---

### 2. **Global Comparison Test**

**Purpose:** Determine if *any* configurations differ significantly from each other.

#### If data is normal → **One-Way ANOVA**
- Tests if means differ across groups
- F-statistic: ratio of between-group variance to within-group variance
- Larger F → more confident differences are real
- **Decision:** If p ≤ 0.05, at least one pair differs significantly

#### If data is not normal → **Kruskal-Wallis H-test**
- Non-parametric equivalent to ANOVA
- Based on rank order (not actual values)
- More robust to outliers and non-normal distributions
- **Decision:** If p ≤ 0.05, at least one pair differs significantly

**Example:** 
- ANOVA: F=5.23, p=0.012 → **Significant** — population sizes ARE NOT all equivalent
- Kruskal-Wallis: H=6.45, p=0.004 → **Significant** — same conclusion via ranks

---

### 3. **Pairwise Comparisons (Post-Hoc Tests)**

After finding global significance, we test *which pairs* differ.

#### If data is normal → **Welch's t-test** (with Bonferroni correction)
- Two-sample t-test not assuming equal variances
- Effect size: **Cohen's d** = (mean₁ − mean₂) / pooled_std
  - d < 0.2: negligible effect
  - d ≈ 0.5: medium effect
  - d > 0.8: large effect

#### If data is not normal → **Mann-Whitney U test** (with Bonferroni correction)
- Non-parametric equivalent to t-test
- Compares distributions using ranks
- Effect size: **Rank-biserial correlation** (−1 to +1)
  - |r| < 0.1: negligible
  - |r| ≈ 0.3: medium
  - |r| > 0.5: large

**Bonferroni Correction:**
For k configurations, there are k(k−1)/2 pairwise comparisons. 
- **Uncorrected α:** 0.05 (prone to false positives)
- **Bonferroni α:** 0.05 / (number of comparisons)
- **Example:** 4 configurations → 6 pairs → α = 0.05/6 = 0.0083

This controls the family-wise error rate (probability of ANY false positive).

**Example Output:**
```
  100 vs 200: Δμ=+0.01234  Welch's t-test  p=0.042  ***
  100 vs 250: Δμ=+0.00856  Welch's t-test  p=0.087  —
```
Only 100 vs 200 is significant at the corrected level.

---

### 4. **Reasoning: Is the Best Config Actually Better?**

Consider population size = 250 has mean RMSE = 0.1923 vs 300 = 0.1927.

**Without statistics:** "250 is better by 0.0004"  
**Problem:** Could this be random noise from 5 trials?

**With statistics:**
1. Test normality → p = 0.23 (normal ✓)
2. ANOVA on all four sizes → p = 0.008 (significant ✓ — sizes DO matter)
3. 250 vs 300 pairwise → Welch's t, p = 0.043, d = 0.45 (medium effect, **just barely significant after Bonferroni** ⚠)
4. Evolution: divergence starts Gen 50 (significant from Gen 50 onward)

**Conclusion:** 
- ✓ Population size **does** affect final RMSE significantly
- ⚠ 250 vs 300 are marginally different (p ≈ 0.05); effect is medium
- ✓ The advantage emerges ~halfway through (Gen 50)
- → **250 is a good choice, but 300 is only slightly worse** (consider runtime cost)

---

### 6. **Practical Interpretation Guide**

| Scenario | Finding | Action |
|----------|---------|--------|
| p > 0.05 (not significant) | No meaningful difference detected | Parameters are equivalent; choose based on speed/simplicity |
| p ≤ 0.05, small effect (d < 0.2) | Statistically significant but practically negligible | Technically better, but may not matter |
| p ≤ 0.05, medium effect (d ≈ 0.5) | Statistically & practically significant | Clear recommendation for better parameter |
| p ≤ 0.05, large effect (d > 0.8) | Strongly significant difference | Strongly recommend best parameter |
