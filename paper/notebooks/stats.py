"""
Academic statistics for honest backtest evaluation.

Implements:
- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
- Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation
- Skewness / kurtosis correction for non-normal returns

These are non-negotiable for any backtest claim that goes into a publication.
Raw Sharpe alone overstates real edge by 1.5-3× in retail settings.

References:
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management, 40(5).
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).
  The Probability of Backtest Overfitting. SSRN 2326253.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# DEFLATED SHARPE RATIO
# =============================================================================

def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio. Returns expected as periodic (daily/4h/etc)."""
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float((r.mean() / r.std()) * np.sqrt(periods_per_year))


def deflated_sharpe(
    sr_observed: float,
    n_trials: int,
    n_observations: int,
    returns_skew: float = 0.0,
    returns_kurt: float = 3.0,
) -> Tuple[float, float]:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Corrects the observed Sharpe for:
    1. Selection bias from testing N strategies
    2. Non-normality of returns (skew, kurt)

    Returns:
        (deflated_sr, probability_real_edge)

    deflated_sr — Sharpe after correction; treat as the realistic estimate.
    probability_real_edge — P(true Sharpe > 0) given observed.

    A backtest with sr_observed = 1.0 from 1000 trials and 252 observations
    has DSR roughly 0.3 — that's the level you should trust live.
    """
    if n_trials < 1 or n_observations < 2:
        return 0.0, 0.0

    # Expected maximum Sharpe under null (no real edge), via order statistics
    # of N independent N(0,1) trials (Bailey & Lopez de Prado 2014, Eq. 7-8).
    euler_gamma = 0.5772156649
    e_max_z = (1 - euler_gamma) * _inv_norm_cdf(1 - 1.0 / n_trials) + euler_gamma * _inv_norm_cdf(
        1 - 1.0 / (n_trials * np.e)
    )

    # Convert to Sharpe space: SR_null_max = E[max Z] / sqrt(T)
    sr_null_max = e_max_z / np.sqrt(n_observations)

    # Standard error of observed Sharpe, with skew/kurt correction
    skew = returns_skew
    kurt = returns_kurt  # excess kurtosis would be (kurt - 3)
    excess_kurt = kurt - 3.0
    sr_std_error = np.sqrt(
        (1 - skew * sr_observed + (excess_kurt / 4.0) * sr_observed**2)
        / (n_observations - 1)
    )

    if sr_std_error <= 0:
        return 0.0, 0.0

    # Test statistic
    z = (sr_observed - sr_null_max) / sr_std_error

    # Probability of real edge = standard normal CDF of z
    prob_real = _norm_cdf(z)

    # The "deflated SR" itself: shift the observed by the null max
    deflated_sr_value = max(0.0, sr_observed - sr_null_max)

    return float(deflated_sr_value), float(prob_real)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2)))


def _inv_norm_cdf(p: float, eps: float = 1e-12) -> float:
    """Inverse of standard normal CDF (Beasley-Springer-Moro approximation)."""
    p = float(np.clip(p, eps, 1 - eps))
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def _erf(x: float) -> float:
    """Numerical erf via Abramowitz approximation 7.1.26 (sufficient for CDF)."""
    a1, a2, a3, a4, a5 = (
        0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    )
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# =============================================================================
# PROBABILITY OF BACKTEST OVERFITTING (PBO)
# =============================================================================

def pbo(returns_matrix: pd.DataFrame, n_splits: int = 16) -> Tuple[float, dict]:
    """
    Probability of Backtest Overfitting via Combinatorially Symmetric
    Cross-Validation (Bailey, Borwein, López de Prado, Zhu 2014).

    Args:
        returns_matrix: DataFrame where each column is one strategy's
                        time-series of returns, rows are time periods.
                        At least 4 strategies and 50+ rows recommended.
        n_splits: Number of CSCV partitions (must be even, default 16).

    Returns:
        (pbo_score, details)
            pbo_score in [0, 1]. PBO > 0.5 means the in-sample ranking
            of strategies does NOT predict out-of-sample ranking — i.e.
            the experiment is overfit; "best in-sample" is essentially random.

            details is a dict with intermediate stats.

    Interpretation:
        PBO ≈ 0.0 → ranking is stable, in-sample winners are real
        PBO ≈ 0.5 → coin flip; in-sample winners are noise
        PBO ≈ 1.0 → in-sample winners systematically WORST out-of-sample
                    (extreme overfitting / data snooping inversion)
    """
    M = returns_matrix.dropna()
    T, N = M.shape
    if N < 2:
        return 0.0, {"error": "need ≥2 strategies"}
    if T < 4:
        return 0.0, {"error": f"need ≥4 obs; got {T}"}
    # Adapt n_splits to T. For small T, use T directly (each block = 1 row).
    if n_splits % 2 != 0:
        n_splits += 1
    if n_splits > T:
        # Use largest even number ≤ T, with floor at 4
        n_splits = max(4, (T // 2) * 2)
    # Warn (informational) when sample is small
    small_sample_warning = None
    if T < 10:
        small_sample_warning = (
            f"small sample (T={T} < 10): PBO estimate has low statistical power"
        )

    # Split T rows into n_splits roughly-equal partitions
    bounds = np.linspace(0, T, n_splits + 1, dtype=int)
    blocks = [M.iloc[bounds[i]:bounds[i + 1]] for i in range(n_splits)]

    # All ways to choose n_splits/2 of n_splits blocks for "in-sample" half
    half = n_splits // 2
    logits = []
    for is_blocks in combinations(range(n_splits), half):
        is_set = set(is_blocks)
        is_data = pd.concat([blocks[i] for i in is_set])
        oos_data = pd.concat([blocks[i] for i in range(n_splits) if i not in is_set])

        is_sharpes = is_data.apply(lambda c: sharpe_ratio(c, periods_per_year=252))
        oos_sharpes = oos_data.apply(lambda c: sharpe_ratio(c, periods_per_year=252))

        # Best in-sample strategy
        best_is = is_sharpes.idxmax()
        # Its rank in OOS
        oos_rank = oos_sharpes.rank(ascending=True)[best_is]
        # Normalized rank in [0, 1]
        omega = (oos_rank - 1) / (N - 1) if N > 1 else 0.5
        # Logit (avoiding log(0))
        omega_clipped = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logit = np.log(omega_clipped / (1 - omega_clipped))
        logits.append(logit)

    logits = np.array(logits)
    pbo_score = float(np.mean(logits < 0))
    details = {
        "n_splits": n_splits,
        "n_strategies": N,
        "n_observations": T,
        "logits_mean": float(logits.mean()),
        "logits_std": float(logits.std()),
        "n_combinations": len(logits),
    }
    if small_sample_warning is not None:
        details["warning"] = small_sample_warning
    return pbo_score, details


# =============================================================================
# HELPER: report results table with all corrections
# =============================================================================

def evaluate_strategy(
    returns: pd.Series,
    n_trials_estimated: int = 100,
    periods_per_year: int = 252,
    label: str = "",
) -> dict:
    """One-stop: raw Sharpe + deflated Sharpe + distribution stats.

    n_trials_estimated should reflect HONEST count of strategies you
    tested before reporting. Including hidden parameter trials, this is
    usually 100-1000 for retail builds (each parameter combo = one trial).
    """
    r = returns.dropna()
    if len(r) < 2:
        return {"label": label, "n": len(r), "error": "insufficient data"}

    sr = sharpe_ratio(r, periods_per_year=periods_per_year)
    skew = float(r.skew())
    kurt = float(r.kurt() + 3)  # pandas .kurt() returns excess kurt
    dsr, prob = deflated_sharpe(
        sr_observed=sr,
        n_trials=n_trials_estimated,
        n_observations=len(r),
        returns_skew=skew,
        returns_kurt=kurt,
    )
    return {
        "label": label,
        "n_obs": len(r),
        "sharpe_raw": round(sr, 3),
        "sharpe_deflated": round(dsr, 3),
        "prob_real_edge": round(prob, 3),
        "skew": round(skew, 3),
        "kurt": round(kurt, 3),
        "annualized_ret_pct": round(r.mean() * periods_per_year * 100, 2),
    }
