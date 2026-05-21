"""
Statistical Analysis Module for GA Results

Provides functions to:
1. Test normality of RMSE distributions (Shapiro-Wilk test)
2. Compare groups with appropriate statistical tests:
   - Parametric (normal): ANOVA + pairwise t-tests with Bonferroni correction
   - Non-parametric (not normal): Kruskal-Wallis + pairwise Mann-Whitney U tests
3. Analyze both final RMSE and evolution trajectories
4. Identify statistically significant differences between configurations
"""

import json
import numpy as np
import re
from pathlib import Path
from scipy import stats
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass


@dataclass
class NormalityTest:
    """Results of normality testing"""
    statistic: float
    p_value: float
    is_normal: bool
    test_name: str = "Shapiro-Wilk"
    
    def __str__(self):
        status = "✓ Normal" if self.is_normal else "✗ Not Normal"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f} {status}"


@dataclass
class ComparisonResult:
    """Results of statistical comparison tests"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    effect_size: float = None  # Cohen's d or rank-biserial
    
    def __str__(self):
        sig = "✓ Significant" if self.is_significant else "✗ Not Significant"
        effect_str = f", effect_size={self.effect_size:.3f}" if self.effect_size else ""
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4f} {sig}{effect_str}"


class GeneticAlgorithmStatisticalAnalysis:
    """
    Comprehensive statistical analysis of GA results from JSON files.
    
    Workflow:
    1. Load results from a folder (one subfolder per parameter value)
    2. Test normality of final RMSE values
    3. Apply appropriate comparative tests (parametric or non-parametric)
    4. Provide detailed summaries
    """
    
    def __init__(self, results_dir: Path, alpha: float = 0.05):
        """
        Parameters
        ----------
        results_dir : Path
            Directory containing result JSON files
        alpha : float
            Significance level for all statistical tests (default: 0.05)
        """
        self.results_dir = Path(results_dir)
        self.alpha = alpha
        self.data = {}  # param_value → {'histories': [], 'finals': []}
        self.normality_tests = {}  # param_value → NormalityTest
        self.comparison_results = {}  # test_name → ComparisonResult
        
    def load_results(self) -> None:
        """Load all JSON files from results_dir, grouped by parameter value."""
        json_files = sorted(self.results_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.results_dir}")
        
        print(f"Loading {len(json_files)} JSON files from {self.results_dir.name}/")
        
        for json_file in json_files:
            # --- NEW: Filter out ignored files ---
            # Ignores files starting with a capital letter, or files that contain "diversity"
            if json_file.name[0].isupper() or 'diversity' in json_file.name.lower():
                continue
            # -------------------------------------

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract parameter value from filename or pipeline name
                pipeline = data.get('pipeline', '')
                param_value = self._extract_param_value(json_file.stem, pipeline)
                
                if param_value not in self.data:
                    self.data[param_value] = {'histories': [], 'finals': []}
                
                # Extract RMSE data
                history = data['results']['history']
                final = data['results']['best_fitness']
                
                self.data[param_value]['histories'].append(history)
                self.data[param_value]['finals'].append(final)
                
            except Exception as e:
                print(f"  ⚠ Skipped {json_file.name}: {e}")
        
        # Sort parameter values
        self.data = dict(sorted(self.data.items()))
        
        print(f"\n✓ Loaded data for {len(self.data)} parameter values:")
        for param_val, d in self.data.items():
            print(f"    {param_val}: {len(d['finals'])} trials")
    
    def _extract_param_value(self, filename: str, pipeline: str) -> str:
        """Extract parameter value from filename or pipeline name."""
        
        # 1. Remove the timestamp suffix (e.g., _2026-05-16T21-42-06...)
        # This regex looks for an underscore followed by a 4-digit year and removes it and everything after.
        base_name = re.sub(r'_\d{4}-\d{2}-\d{2}T.*$', '', filename)
        
        # Convert to lowercase just in case there are slight casing mismatches
        base_name_lower = base_name.lower()
        
        # 2. Extract values based on your specific prefixes
        if base_name_lower.startswith('selectiontype-'):
            return base_name[len('selectiontype-'):]  # Returns 'ranking', 'roulette', etc.
            
        elif base_name_lower.startswith('crossover-'):
            return base_name[len('crossover-'):]      # Returns 'cycle-r0.9', 'single_point_2children-r0.9', etc.
            
        elif base_name_lower.startswith('fitnesssharing-'):
            return base_name[len('fitnesssharing-'):] # Returns 'baseline', 'sigma0.1', etc.
            
        elif base_name_lower.startswith('restrictedmating-'):
            return base_name[len('restrictedmating-'):] # Returns 'baseline', 'best_partial_match', etc.
    
        elif base_name_lower.startswith('alpha-'):
            return base_name[len('alpha-'):] # Returns '5-255', '128-255', etc.

        # 3. Fallback for simple numeric parameters (population_size, elitism, mutation_rate)
        # E.g., "popsize-100" -> "100", "mutation-rate-0.05" -> "0.05"
        parts = base_name.split('-')
        for part in reversed(parts):
            if part.replace('.', '', 1).isdigit():
                return part
        
        return base_name
    
    def test_normality(self) -> Dict[str, NormalityTest]:
        """
        Test normality of final RMSE distributions for each parameter value.
        Uses Shapiro-Wilk test (good for small samples like n=5).
        
        Returns
        -------
        dict
            Mapping of parameter_value → NormalityTest
        """
        print("\n" + "="*70)
        print("NORMALITY TESTING — Shapiro-Wilk Test (H0: data is normally distributed)")
        print("="*70)
        
        for param_val, d in self.data.items():
            finals = np.array(d['finals'])
            
            # Shapiro-Wilk test
            stat, p_value = stats.shapiro(finals)
            is_normal = p_value > self.alpha
            
            self.normality_tests[param_val] = NormalityTest(
                statistic=stat,
                p_value=p_value,
                is_normal=is_normal
            )
            
            status = "Normal     ✓" if is_normal else "Not Normal ✗"
            print(f"  {str(param_val):>6}: {status}  "
                  f"W={stat:.4f}, p={p_value:.4f}  (n={len(finals)})")
        
        return self.normality_tests
    
    def _are_all_normal(self) -> bool:
        """Check if all groups are normally distributed."""
        return all(test.is_normal for test in self.normality_tests.values())
    
    def compare_final_rmse(self) -> Dict[str, any]:
        """
        Compare final RMSE distributions across all parameter values.
        
        Parametric (if normal): ANOVA + Tukey HSD pairwise comparisons
        Non-parametric (if not normal): Kruskal-Wallis + Mann-Whitney U pairwise tests
        
        Returns
        -------
        dict
            Contains:
            - 'omnibus': result of global comparison test
            - 'pairwise': pairwise comparison results
            - 'best_param': best parameter value with lowest RMSE
            - 'summary': DataFrame with descriptive statistics
        """
        print("\n" + "="*70)
        print("COMPARING FINAL RMSE VALUES")
        print("="*70)
        
        # Prepare data
        all_finals = []
        param_labels = []
        for param_val, d in self.data.items():
            finals = d['finals']
            all_finals.extend(finals)
            param_labels.extend([param_val] * len(finals))
        
        all_finals = np.array(all_finals)
        param_labels = np.array(param_labels)
        
        # Descriptive statistics
        summary_data = []
        for param_val in self.data.keys():
            finals = np.array(self.data[param_val]['finals'])
            summary_data.append({
                'Parameter': param_val,
                'Mean RMSE': finals.mean(),
                'Std Dev': finals.std(),
                'Min': finals.min(),
                'Max': finals.max(),
                'N': len(finals)
            })
        summary_df = pd.DataFrame(summary_data)
        
        print("\nDescriptive Statistics:")
        print(summary_df.to_string(index=False))
        
        # Check if we have at least 2 groups for comparison
        if len(self.data) < 2:
            print("\n⚠️  Only 1 parameter value found. Cannot perform statistical comparison.")
            print("   (Need at least 2 groups for comparison tests)")
            
            # Identify best configuration
            best_param = min(self.data.keys(), 
                            key=lambda p: np.mean(self.data[p]['finals']))
            best_rmse = np.mean(self.data[best_param]['finals'])
            
            print(f"\n🏆 Best (and only) configuration: {best_param} (mean RMSE = {best_rmse:.6f})")
            
            return {
                'omnibus': None,
                'pairwise': [],
                'best_param': best_param,
                'summary': summary_df
            }
        
        # Global comparison test
        is_normal = self._are_all_normal()
        
        if is_normal:
            print("\n✓ Data is normally distributed → Using ANOVA (parametric)")
            result = self._compare_final_rmse_anova(all_finals, param_labels)
        else:
            print("\n✗ Data is not normally distributed → Using Kruskal-Wallis (non-parametric)")
            result = self._compare_final_rmse_kruskal_wallis()
        
        # Pairwise comparisons
        print("\nPairwise Comparisons:")
        pairwise = self._pairwise_comparisons_final(is_normal)

        # Compute means and identify the param with minimal mean RMSE
        means = {p: np.mean(self.data[p]['finals']) for p in self.data.keys()}
        best_param = min(means, key=means.get)
        best_rmse = means[best_param]

        # Build a lookup for pairwise results for quick access
        pair_map = {}
        for r in pairwise:
            pair_map[(r['Param1'], r['Param2'])] = r

        # Determine contenders: configs not shown to be significantly worse than the best
        contenders = set([best_param])
        for other in self.data.keys():
            if other == best_param:
                continue

            # Find pair result regardless of ordering
            key = (best_param, other) if (best_param, other) in pair_map else (other, best_param)
            r = pair_map.get(key)
            if r is None:
                # missing pair info: conservatively include as contender
                contenders.add(other)
                continue

            mean_diff = r['Mean_Diff']
            is_sig = r['Is_Significant']

            # Determine if 'best_param' is significantly better than 'other'
            if key[0] == best_param:
                # r corresponds to (best, other): mean_diff = mean(best)-mean(other)
                best_better = (mean_diff < 0) and is_sig
            else:
                # r corresponds to (other, best): mean_diff = mean(other)-mean(best)
                best_better = (mean_diff > 0) and is_sig

            # If best is NOT significantly better, include other as contender
            if not best_better:
                contenders.add(other)

        is_unique_best = (len(contenders) == 1)

        # Print summary
        if is_unique_best:
            print(f"\n🏆 Unique best configuration: {best_param} (mean RMSE = {best_rmse:.6f}) — significantly better than all others")
        else:
            print(f"\n⚠ No single statistically superior configuration found.")
            print(f"  Mean of best candidate: {best_param} = {best_rmse:.6f}")
            print(f"  Contenders (not proven worse than best): {sorted(contenders)}")

        # Save results
        self.comparison_results = {
            'omnibus': result,
            'pairwise': pairwise,
            'best_param': best_param,
            'best_rmse': best_rmse,
            'is_unique_best': is_unique_best,
            'contenders': sorted(contenders),
            'summary': summary_df
        }
        return self.comparison_results
    
    def _compare_final_rmse_anova(self, all_finals: np.ndarray, 
                                   param_labels: np.ndarray) -> ComparisonResult:
        """One-way ANOVA for normally distributed data."""
        groups = [np.array(self.data[p]['finals']) for p in sorted(self.data.keys())]
        f_stat, p_value = stats.f_oneway(*groups)
        
        is_significant = p_value < self.alpha
        
        result = ComparisonResult(
            test_name="One-Way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha
        )
        
        print(f"\n{result}")
        
        return result
    
    def _compare_final_rmse_kruskal_wallis(self) -> ComparisonResult:
        """Kruskal-Wallis test for non-normally distributed data."""
        # Convert all groups to numpy arrays explicitly
        groups = [np.array(self.data[p]['finals'], dtype=np.float64) 
                  for p in sorted(self.data.keys())]
        
        h_stat, p_value = stats.kruskal(*groups)
        
        is_significant = p_value < self.alpha
        
        result = ComparisonResult(
            test_name="Kruskal-Wallis H-test",
            statistic=h_stat,
            p_value=p_value,
            is_significant=is_significant,
            alpha=self.alpha
        )
        
        print(f"\n{result}")
        
        return result
    
    def _pairwise_comparisons_final(self, is_normal: bool) -> Dict:
        """
        Pairwise comparisons with multiple testing correction.
        
        Parametric: t-tests with Bonferroni correction
        Non-parametric: Mann-Whitney U tests with Bonferroni correction
        """
        param_values = sorted(self.data.keys())
        n_comparisons = len(param_values) * (len(param_values) - 1) // 2
        bonferroni_alpha = self.alpha / n_comparisons
        
        print(f"  (Bonferroni-corrected α = {bonferroni_alpha:.6f} for {n_comparisons} comparisons)")
        
        pairwise_results = []
        
        for i, p1 in enumerate(param_values):
            for p2 in param_values[i+1:]:
                group1 = np.array(self.data[p1]['finals'], dtype=np.float64)
                group2 = np.array(self.data[p2]['finals'], dtype=np.float64)
                
                if is_normal:
                    # Welch's t-test (doesn't assume equal variances)
                    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    test_name = "Welch's t-test"
                    effect_size = (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)
                else:
                    # Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    t_stat = u_stat
                    test_name = "Mann-Whitney U"
                    # Rank-biserial correlation as effect size
                    n1, n2 = len(group1), len(group2)
                    effect_size = 1 - (2*u_stat) / (n1 * n2)
                
                is_sig = p_value < bonferroni_alpha
                
                mean_diff = group1.mean() - group2.mean()
                sig_marker = "***" if is_sig else "—"
                
                print(f"    {p1:>6} vs {p2:>6}: Δμ={mean_diff:+.6f}  {test_name}  "
                      f"p={p_value:.6f}  {sig_marker}")
                
                pairwise_results.append({
                    'Param1': p1,
                    'Param2': p2,
                    'Mean_Diff': mean_diff,
                    'P_Value': p_value,
                    'Is_Significant': is_sig,
                    'Effect_Size': effect_size
                })
        
        return pairwise_results
    
    pass


def analyze_ga_results(results_dir: Path, alpha: float = 0.05) -> GeneticAlgorithmStatisticalAnalysis:
    """
    Convenience function to run complete statistical analysis pipeline.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing result JSON files
    alpha : float
        Significance level (default: 0.05)
    
    Returns
    -------
    GeneticAlgorithmStatisticalAnalysis
        Initialized and analyzed object with all results
    
    Example
    -------
    >>> analysis = analyze_ga_results(project_root / "results" / "population_size")
    >>> # All tests run automatically
    """
    analyzer = GeneticAlgorithmStatisticalAnalysis(results_dir, alpha=alpha)
    analyzer.load_results()
    analyzer.test_normality()
    analyzer.compare_final_rmse()
    return analyzer
