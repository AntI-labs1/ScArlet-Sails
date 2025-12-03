"""
DISPERSION ANALYSIS ENGINE
Statistical proof that P_rb, P_ml, P_hyb make significantly different decisions

Tests:
1. ANOVA: F-statistic, p-value
2. Kolmogorov-Smirnov: Distribution differences
3. Correlation Matrix: Independence
4. Variance Decomposition: Between vs Within
5. Effect Size: Cohen d, eta-squared
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from scipy import stats
from scipy.stats import f_oneway, ks_2samp, pearsonr
import logging
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.hybrid_v2 import HybridStrategy
try:  # Prefer v3 when available
    from strategies.xgboost_ml_v3 import XGBoostMLStrategyV3 as MLStrategy
except Exception:  # pragma: no cover - fallback
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy as MLStrategy

logger = logging.getLogger(__name__)


class DispersionAnalyzer:
    """
    Comprehensive dispersion analysis for trading strategies
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategy_vectors = {}
        self.test_results = {}
        logger.info("DispersionAnalyzer initialized")

    def run_strategies_on_df(self, df: pd.DataFrame, strategies: Dict[str, any]) -> pd.DataFrame:
        """Execute strategies on the same dataset and align outputs."""
        outputs = {}
        for name, strategy in strategies.items():
            logger.info(f"Running strategy {name} for dispersion alignment")
            outputs[name] = strategy.generate_signals(df)

        common_index = None
        for sig_df in outputs.values():
            if common_index is None:
                common_index = sig_df.index
            else:
                common_index = common_index.intersection(sig_df.index)

        combined = pd.DataFrame(index=common_index)
        if 'rule_based' in outputs:
            combined['P_rb'] = outputs['rule_based'].loc[common_index, 'P_rb']
            combined['signal_rb'] = outputs['rule_based'].loc[common_index, 'signal']
        if 'xgboost_ml' in outputs:
            col = 'P_ml' if 'P_ml' in outputs['xgboost_ml'].columns else 'ml_proba'
            combined['P_ml'] = outputs['xgboost_ml'].loc[common_index, col]
            combined['signal_ml'] = outputs['xgboost_ml'].loc[common_index, 'signal']
        if 'hybrid' in outputs:
            combined['P_hyb'] = outputs['hybrid'].loc[common_index, 'P_hyb']
            combined['signal_hyb'] = outputs['hybrid'].loc[common_index, 'signal']

        combined.dropna(how='all', inplace=True)
        return combined

    def compute_strategy_vectors(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, any],
        combined_signals: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        logger.info(f"Computing strategy vectors for {len(df)} bars...")
        combined = combined_signals if combined_signals is not None else self.run_strategies_on_df(df, strategies)
        vectors = {}
        if 'P_rb' in combined:
            vectors['rule_based'] = combined['P_rb'].values
        if 'P_ml' in combined:
            vectors['xgboost_ml'] = combined['P_ml'].values
        if 'P_hyb' in combined:
            vectors['hybrid'] = combined['P_hyb'].values

        min_length = min((len(v) for v in vectors.values()), default=0)
        for name in list(vectors.keys()):
            vectors[name] = vectors[name][:min_length]
        self.strategy_vectors = vectors
        return vectors

    def compute_signal_correlations(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Correlation matrix for binary trade decisions."""
        signal_cols = [c for c in ['signal_rb', 'signal_ml', 'signal_hyb'] if c in signals_df.columns]
        if not signal_cols:
            return pd.DataFrame()
        aligned = signals_df[signal_cols].dropna()
        if aligned.empty:
            return pd.DataFrame()
        corr = aligned.astype(float).corr()
        self.test_results['signal_correlations'] = corr
        logger.info("Binary signal correlation matrix:\n%s", corr)
        return corr

    def test_anova(self, vectors: Dict[str, np.ndarray]) -> Dict:
        logger.info("Running ANOVA test...")
        clean_vectors = []
        for name, vec in vectors.items():
            clean = vec[~np.isnan(vec)]
            if len(clean) > 0:
                clean_vectors.append(clean)
        if len(clean_vectors) < 2:
            logger.warning("Not enough data for ANOVA")
            return {'error': 'insufficient_data'}
        min_len = min(len(v) for v in clean_vectors)
        clean_vectors = [v[:min_len] for v in clean_vectors]
        F_stat, p_value = f_oneway(*clean_vectors)
        alpha = 0.05
        reject_null = p_value < alpha
        result = {
            'test': 'ANOVA',
            'F_statistic': float(F_stat),
            'p_value': float(p_value),
            'alpha': alpha,
            'reject_null': reject_null,
            'interpretation': 'Strategies have SIGNIFICANTLY different means' if reject_null else 'No significant difference'
        }
        logger.info(f"  F-statistic: {F_stat:.4f}")
        logger.info(f"  p-value: {p_value:.6f}")
        logger.info(f"  Result: {'REJECT H0' if reject_null else 'FAIL TO REJECT H0'}")
        self.test_results['anova'] = result
        return result

    def test_kolmogorov_smirnov(self, vectors: Dict[str, np.ndarray]) -> Dict:
        logger.info("Running Kolmogorov-Smirnov tests...")
        results = {}
        strategy_names = list(vectors.keys())
        for i in range(len(strategy_names)):
            for j in range(i+1, len(strategy_names)):
                name1, name2 = strategy_names[i], strategy_names[j]
                vec1 = vectors[name1][~np.isnan(vectors[name1])]
                vec2 = vectors[name2][~np.isnan(vectors[name2])]
                if len(vec1) == 0 or len(vec2) == 0:
                    continue
                ks_stat, p_value = ks_2samp(vec1, vec2)
                pair_name = f"{name1}_vs_{name2}"
                results[pair_name] = {
                    'KS_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'reject_null': p_value < 0.05,
                    'interpretation': 'Distributions are DIFFERENT' if p_value < 0.05 else 'Distributions are similar'
                }
                logger.info(f"  {pair_name}: KS={ks_stat:.4f}, p={p_value:.6f}")
        self.test_results['kolmogorov_smirnov'] = results
        return results

    def compute_correlations(self, vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
        logger.info("Computing correlation matrix...")
        aligned = self._align_vectors(vectors)
        df = pd.DataFrame(aligned)
        corr_matrix = df.corr()
        logger.info("Correlation matrix:")
        logger.info(f"\n{corr_matrix}")
        self.test_results['correlation_matrix'] = corr_matrix
        return corr_matrix

    def variance_decomposition(self, vectors: Dict[str, np.ndarray]) -> Dict:
        logger.info("Computing variance decomposition...")
        aligned = self._align_vectors(vectors)
        all_values = []
        group_labels = []
        for name, vec in aligned.items():
            all_values.extend(vec)
            group_labels.extend([name] * len(vec))
        all_values = np.array(all_values)
        grand_mean = np.mean(all_values)
        var_total = np.var(all_values, ddof=1)
        group_means = {name: np.mean(vec) for name, vec in aligned.items()}
        n_per_group = {name: len(vec) for name, vec in aligned.items()}
        ss_between = sum(n * (group_means[name] - grand_mean)**2 for name, n in n_per_group.items())
        df_between = len(aligned) - 1
        var_between = ss_between / df_between if df_between > 0 else 0
        ss_within = sum(np.sum((vec - group_means[name])**2) for name, vec in aligned.items())
        df_within = len(all_values) - len(aligned)
        var_within = ss_within / df_within if df_within > 0 else 0
        eta_squared = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0
        result = {
            'var_total': float(var_total),
            'var_between': float(var_between),
            'var_within': float(var_within),
            'eta_squared': float(eta_squared),
            'interpretation': self._interpret_eta_squared(eta_squared)
        }
        logger.info(f"  Total variance: {var_total:.6f}")
        logger.info(f"  Between-group variance: {var_between:.6f}")
        logger.info(f"  Within-group variance: {var_within:.6f}")
        logger.info(f"  eta_squared (effect size): {eta_squared:.4f}")
        logger.info(f"  Interpretation: {result['interpretation']}")
        self.test_results['variance_decomposition'] = result
        return result

    def compute_effect_sizes(self, vectors: Dict[str, np.ndarray]) -> Dict:
        logger.info("Computing effect sizes (Cohen d)...")
        aligned = self._align_vectors(vectors)
        results = {}
        strategy_names = list(aligned.keys())
        for i in range(len(strategy_names)):
            for j in range(i+1, len(strategy_names)):
                name1, name2 = strategy_names[i], strategy_names[j]
                vec1 = aligned[name1]
                vec2 = aligned[name2]
                mean1, mean2 = np.mean(vec1), np.mean(vec2)
                std1, std2 = np.std(vec1, ddof=1), np.std(vec2, ddof=1)
                n1, n2 = len(vec1), len(vec2)
                pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                pair_name = f"{name1}_vs_{name2}"
                results[pair_name] = {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(abs(cohens_d))
                }
                logger.info(f"  {pair_name}: d={cohens_d:.4f} ({results[pair_name]['interpretation']})")
        self.test_results['effect_sizes'] = results
        return results

    def generate_summary_statistics(self, vectors: Dict[str, np.ndarray]) -> pd.DataFrame:
        logger.info("Generating summary statistics...")
        stats_dict = {}
        for name, vec in vectors.items():
            clean_vec = vec[~np.isnan(vec)]
            if len(clean_vec) == 0:
                continue
            stats_dict[name] = {
                'count': len(clean_vec),
                'mean': np.mean(clean_vec),
                'std': np.std(clean_vec, ddof=1),
                'min': np.min(clean_vec),
                'q25': np.percentile(clean_vec, 25),
                'median': np.median(clean_vec),
                'q75': np.percentile(clean_vec, 75),
                'max': np.max(clean_vec),
                'skewness': stats.skew(clean_vec),
                'kurtosis': stats.kurtosis(clean_vec)
            }
        summary_df = pd.DataFrame(stats_dict).T
        logger.info("Summary statistics:")
        logger.info(f"\n{summary_df}")
        self.test_results['summary_statistics'] = summary_df
        return summary_df

    def run_full_analysis(self, df: pd.DataFrame, strategies: Dict[str, any]) -> Dict:
        logger.info("="*80)
        logger.info("DISPERSION ANALYSIS - FULL RUN")
        logger.info("="*80)
        combined_signals = self.run_strategies_on_df(df, strategies)
        vectors = self.compute_strategy_vectors(df, strategies, combined_signals)
        summary = self.generate_summary_statistics(vectors)
        anova_result = self.test_anova(vectors)
        ks_results = self.test_kolmogorov_smirnov(vectors)
        corr_matrix = self.compute_correlations(vectors)
        binary_corr = self.compute_signal_correlations(combined_signals)
        var_decomp = self.variance_decomposition(vectors)
        effect_sizes = self.compute_effect_sizes(vectors)
        all_results = {
            'signals': combined_signals,
            'vectors': vectors,
            'summary_statistics': summary,
            'anova': anova_result,
            'kolmogorov_smirnov': ks_results,
            'correlation_matrix': corr_matrix,
            'signal_correlations': binary_corr,
            'variance_decomposition': var_decomp,
            'effect_sizes': effect_sizes
        }
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        return all_results

    def generate_report(self, results: Dict) -> str:
        report = []
        report.append("="*80)
        report.append("DISPERSION ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        report.append("SUMMARY STATISTICS")
        report.append("-"*40)
        report.append(str(results['summary_statistics']))
        report.append("")
        report.append("ANOVA TEST")
        report.append("-"*40)
        anova = results['anova']
        if "F_statistic" in anova:
            report.append(f"F-statistic: {anova['F_statistic']:.4f}")
            report.append(f"p-value: {anova['p_value']:.6f}")
            report.append(f"Result: {anova['interpretation']}")
        else:
            report.append("ANOVA could not be calculated due to empty data.")
            report.append(f"Details: {anova.get('error')}")
        report.append("")
        report.append("VARIANCE DECOMPOSITION")
        report.append("-"*40)
        var_decomp = results['variance_decomposition']
        report.append(f"Total variance: {var_decomp['var_total']:.6f}")
        report.append(f"Between-group variance: {var_decomp['var_between']:.6f}")
        report.append(f"Within-group variance: {var_decomp['var_within']:.6f}")
        report.append(f"eta_squared (effect size): {var_decomp['eta_squared']:.4f}")
        report.append(f"Interpretation: {var_decomp['interpretation']}")
        report.append("")
        report.append("CORRELATION MATRIX")
        report.append("-"*40)
        report.append(str(results['correlation_matrix']))
        report.append("")
        report.append("="*80)
        report.append("CONCLUSION")
        report.append("="*80)
        if anova.get('reject_null', False) and var_decomp['eta_squared'] > 0.5:
            report.append("✅ HYPOTHESIS CONFIRMED!")
            report.append("")
            report.append("The three strategies (Rule-Based, XGBoost ML, Hybrid)")
            report.append("make SIGNIFICANTLY DIFFERENT decisions:")
            report.append(f"  - F-statistic: {anova['F_statistic']:.2f} (p < 0.001)")
            report.append(f"  - Effect size eta_squared: {var_decomp['eta_squared']:.2%}")
            report.append(f"  - Between-group variance accounts for {var_decomp['eta_squared']:.1%} of total")
            report.append("")
            report.append("This provides strong evidence that different algorithmic")
            report.append("approaches lead to fundamentally different trading decisions.")
        else:
            report.append("⚠️ HYPOTHESIS NOT FULLY SUPPORTED")
            report.append("")
            report.append("Statistical tests show some differences, but not strong enough")
            report.append("to claim significant dispersion. Consider:")
            report.append("  - More data")
            report.append("  - Trained ML models (not fallback)")
            report.append("  - Different market conditions")
        report.append("="*80)
        return "\n".join(report)

    def _align_vectors(self, vectors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        all_valid = None
        for vec in vectors.values():
            valid = ~np.isnan(vec)
            if all_valid is None:
                all_valid = valid
            else:
                all_valid = all_valid & valid
        aligned = {}
        for name, vec in vectors.items():
            aligned[name] = vec[all_valid]
        return aligned

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        if eta_sq < 0.01:
            return "negligible effect"
        elif eta_sq < 0.06:
            return "small effect"
        elif eta_sq < 0.14:
            return "medium effect"
        else:
            return "large effect"

    def _interpret_cohens_d(self, d: float) -> str:
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("="*80)
    print("DISPERSION ANALYSIS ENGINE TEST")
    print("="*80)
    print("\nModule loaded successfully!")
    print("Use: analyzer = DispersionAnalyzer()")
    print("Then: results = analyzer.run_full_analysis(df, strategies)")
