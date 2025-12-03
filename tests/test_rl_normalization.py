import numpy as np
import pandas as pd

from rl.trading_environment import normalize_price_features, normalize_volume_features


def test_normalize_price_features_no_leak():
    prices = pd.Series([100, 101, 102, 103])
    feats = normalize_price_features(prices)
    assert set(feats.keys()) == {'latest_ret', 'mean_ret', 'vol', 'trend', 'rel_price'}
    assert np.isfinite(list(feats.values())).all()
    assert abs(feats['trend'] - (103 - 100) / 100) < 1e-6


def test_normalize_volume_features_zscore():
    vols = pd.Series([10, 12, 14, 16])
    feats = normalize_volume_features(vols)
    assert set(feats.keys()) == {'z_vol', 'vol_mean', 'vol_std'}
    assert np.isfinite(list(feats.values())).all()
    assert feats['vol_std'] > 0
    # last point is highest so z-score should be positive
    assert feats['z_vol'] > 0
