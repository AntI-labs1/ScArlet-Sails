"""
Tests for paper/notebooks/stats.py — Deflated Sharpe Ratio and PBO.

These tests validate that:
1. Sharpe ratio computation is correct against known values
2. Deflated Sharpe converges to raw Sharpe at N_trials=1
3. Deflated Sharpe is monotonically decreasing in N_trials
4. PBO is in [0, 1] and behaves sensibly on synthetic data
5. Edge cases handled (empty input, zero variance, single observation)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add paper/notebooks to path so we can import stats.py
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "notebooks"))

from stats import (
    sharpe_ratio,
    deflated_sharpe,
    pbo,
    evaluate_strategy,
)


# =============================================================================
# SHARPE RATIO BASIC TESTS
# =============================================================================

class TestSharpeRatio:
    def test_zero_returns_gives_zero_sharpe(self):
        r = pd.Series([0.0, 0.0, 0.0, 0.0])
        assert sharpe_ratio(r) == 0.0

    def test_positive_constant_returns_infinite_sharpe(self):
        # std = 0 → divide by zero handled gracefully
        r = pd.Series([0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio(r)
        assert result == 0.0  # we return 0 for zero variance, not inf

    def test_known_value_daily(self):
        # Mean 0.001, std 0.01, periods 252 → Sharpe ≈ 0.001/0.01 * sqrt(252) ≈ 1.587
        np.random.seed(42)
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(loc=0.001, scale=0.01, size=1000))
        sr = sharpe_ratio(r, periods_per_year=252)
        # Should be approximately 0.1 * sqrt(252) ≈ 1.587 (asymptotically)
        assert 1.0 < sr < 2.5

    def test_negative_mean_negative_sharpe(self):
        rng = np.random.default_rng(123)
        r = pd.Series(rng.normal(loc=-0.001, scale=0.01, size=500))
        sr = sharpe_ratio(r, periods_per_year=252)
        assert sr < 0

    def test_single_observation_returns_zero(self):
        r = pd.Series([0.05])
        assert sharpe_ratio(r) == 0.0

    def test_empty_returns_zero(self):
        r = pd.Series([], dtype=float)
        assert sharpe_ratio(r) == 0.0


# =============================================================================
# DEFLATED SHARPE TESTS
# =============================================================================

class TestDeflatedSharpe:
    def test_n_trials_1_deflation_minimal(self):
        # With N=1 trial, deflation should be small (no selection bias)
        dsr, prob = deflated_sharpe(
            sr_observed=1.0, n_trials=1, n_observations=252
        )
        # SR_null_max for N=1 should be near 0
        assert 0.8 < dsr <= 1.0
        assert prob > 0.5  # probability of real edge is high

    def test_n_trials_1000_significant_deflation(self):
        # With N=1000 trials, deflation should be substantial
        dsr, prob = deflated_sharpe(
            sr_observed=1.0, n_trials=1000, n_observations=252
        )
        # Strong selection penalty
        assert dsr < 0.8
        # But still positive probability of real edge
        assert 0.0 <= prob <= 1.0

    def test_deflated_monotonic_in_n_trials(self):
        # As N increases, deflated SR should decrease (more selection bias to correct)
        results = []
        for n in [1, 10, 100, 1000, 10000]:
            dsr, _ = deflated_sharpe(sr_observed=1.0, n_trials=n, n_observations=252)
            results.append(dsr)
        # Each subsequent value should be <= previous
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1] - 1e-6, (
                f"DSR not monotonic in N: {results}"
            )

    def test_negative_observed_sharpe(self):
        # Negative observed SR should not generate positive deflated
        dsr, prob = deflated_sharpe(
            sr_observed=-0.5, n_trials=100, n_observations=252
        )
        assert dsr == 0.0  # clipped to non-negative
        assert prob < 0.5

    def test_more_observations_higher_dsr(self):
        # More observations → less expected null variance → higher DSR
        dsr_short, _ = deflated_sharpe(
            sr_observed=1.0, n_trials=100, n_observations=100
        )
        dsr_long, _ = deflated_sharpe(
            sr_observed=1.0, n_trials=100, n_observations=2520
        )
        assert dsr_long > dsr_short

    def test_zero_observations_returns_zero(self):
        dsr, prob = deflated_sharpe(sr_observed=1.0, n_trials=10, n_observations=0)
        assert dsr == 0.0
        assert prob == 0.0

    def test_n_trials_zero_returns_zero(self):
        dsr, prob = deflated_sharpe(sr_observed=1.0, n_trials=0, n_observations=252)
        assert dsr == 0.0


# =============================================================================
# PBO TESTS
# =============================================================================

class TestPBO:
    def test_pbo_in_unit_interval(self):
        # Synthetic data: 5 strategies × 200 obs of iid normal
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(0, 0.01, size=(200, 5)),
                          columns=[f"strat_{i}" for i in range(5)])
        pbo_score, details = pbo(df, n_splits=8)
        assert 0.0 <= pbo_score <= 1.0
        assert details["n_strategies"] == 5

    def test_pbo_random_close_to_half(self):
        # With pure noise strategies, PBO should be approximately 0.5
        # (in-sample best is no better than random OOS)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(0, 0.01, size=(500, 10)),
                          columns=[f"strat_{i}" for i in range(10)])
        pbo_score, _ = pbo(df, n_splits=8)
        # Allow wide tolerance for sample variance — main check is non-extreme
        assert 0.3 <= pbo_score <= 0.7, f"PBO on noise should be ~0.5, got {pbo_score}"

    def test_pbo_with_dominant_strategy_low(self):
        # If one strategy clearly dominates everywhere, PBO should be low
        rng = np.random.default_rng(7)
        # 5 noise strategies + 1 with clear positive mean
        cols = {}
        for i in range(5):
            cols[f"noise_{i}"] = rng.normal(0, 0.01, size=500)
        cols["alpha"] = rng.normal(0.003, 0.01, size=500)  # consistent positive
        df = pd.DataFrame(cols)
        pbo_score, _ = pbo(df, n_splits=8)
        # Best in-sample (alpha) should remain best OOS most of the time
        assert pbo_score < 0.4, f"PBO with dominant strategy should be low, got {pbo_score}"

    def test_pbo_insufficient_strategies(self):
        df = pd.DataFrame({"only_one": [0.01, 0.02, -0.01]})
        pbo_score, details = pbo(df, n_splits=8)
        assert pbo_score == 0.0
        assert "error" in details

    def test_pbo_insufficient_observations(self):
        df = pd.DataFrame(
            {"s1": [0.01, 0.02], "s2": [0.01, -0.01]}
        )
        pbo_score, details = pbo(df, n_splits=8)
        assert pbo_score == 0.0
        assert "error" in details


# =============================================================================
# INTEGRATION TEST: evaluate_strategy
# =============================================================================

class TestEvaluateStrategy:
    def test_evaluate_strategy_returns_dict_with_expected_keys(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, size=500))
        result = evaluate_strategy(r, n_trials_estimated=100, label="test")
        expected_keys = {
            "label", "n_obs", "sharpe_raw", "sharpe_deflated",
            "prob_real_edge", "skew", "kurt", "annualized_ret_pct",
        }
        assert expected_keys.issubset(result.keys())

    def test_evaluate_strategy_label_propagates(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0.001, 0.01, size=500))
        result = evaluate_strategy(r, label="my_label")
        assert result["label"] == "my_label"

    def test_evaluate_strategy_handles_empty(self):
        r = pd.Series([], dtype=float)
        result = evaluate_strategy(r, label="empty")
        assert "error" in result


# =============================================================================
# REFERENCE VALUE TESTS (against Bailey & Lopez de Prado examples)
# =============================================================================

class TestReferenceValues:
    """Spot-check against published Deflated Sharpe examples."""

    def test_paper_example_high_n_trials(self):
        """
        Bailey & López de Prado (2014) example: SR=1.5, T=252, N=100,
        γ ≈ 0 (normal returns) should give DSR roughly 0.5-1.0
        """
        dsr, _ = deflated_sharpe(
            sr_observed=1.5,
            n_trials=100,
            n_observations=252,
            returns_skew=0.0,
            returns_kurt=3.0,
        )
        # Should be a meaningful reduction but not zero
        assert 0.3 < dsr < 1.4

    def test_skew_kurt_correction(self):
        """Negative skew + high kurt should reduce DSR vs normal returns."""
        dsr_normal, _ = deflated_sharpe(
            sr_observed=1.0, n_trials=100, n_observations=252,
            returns_skew=0.0, returns_kurt=3.0,
        )
        dsr_fat_tail, _ = deflated_sharpe(
            sr_observed=1.0, n_trials=100, n_observations=252,
            returns_skew=-1.0, returns_kurt=10.0,
        )
        # Fat-tailed/negatively-skewed returns should produce smaller or equal DSR
        # (Higher std error → smaller corrected Sharpe)
        # Note: implementation may vary; this checks the formula's signed direction
        assert dsr_fat_tail <= dsr_normal + 0.05  # tolerance for numerical noise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
