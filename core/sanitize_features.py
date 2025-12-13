"""
Canonical Feature Sanitization Module.
Single source of truth for data cleaning before model inference.

Design principles:
1. Hard safety: Remove NaN/inf (critical for XGBoost)
2. Soft hygiene: Clip extreme percentiles
3. Edge case handling: Median-based fallback for near-constant columns

Used by: walk_forward_validation, backtest, live inference
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def sanitize_for_model(
    df: pd.DataFrame,
    feature_names: List[str],
    drop_inf: bool = True,
    drop_nan: bool = True,
    clip_extreme: bool = True,
    clip_percentile: float = 99.9,
    spike_threshold: float = 100.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Sanitize DataFrame for model inference.
    
    Args:
        df: Input DataFrame
        feature_names: List of feature columns to sanitize
        drop_inf: Drop rows with inf/-inf values
        drop_nan: Drop rows with NaN values
        clip_extreme: Clip values beyond percentile thresholds
        clip_percentile: Percentile for clipping (default 99.9)
        spike_threshold: For near-constant columns, clip values > median * threshold
        verbose: Print statistics
        
    Returns:
        Cleaned DataFrame (copy)
    """
    df_clean = df.copy()
    initial_len = len(df_clean)
    
    available = [f for f in feature_names if f in df_clean.columns]
    
    if not available:
        logger.warning("No features found in DataFrame")
        return df_clean
    
    dropped_reasons = {}
    
    # 1. Handle inf values (CRITICAL for XGBoost)
    if drop_inf:
        numeric_cols = df_clean[available].select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            inf_mask = np.isinf(df_clean[numeric_cols]).any(axis=1)
            n_inf = inf_mask.sum()
            if n_inf > 0:
                df_clean = df_clean[~inf_mask]
                dropped_reasons['inf'] = n_inf
    
    # 2. Handle NaN values (CRITICAL for XGBoost)
    if drop_nan:
        nan_mask = df_clean[available].isna().any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            df_clean = df_clean[~nan_mask]
            dropped_reasons['nan'] = n_nan
    
    # 3. Clip extreme values
    if clip_extreme and len(df_clean) > 0:
        lower_pct = 100 - clip_percentile
        upper_pct = clip_percentile
        
        for col in available:
            if col not in df_clean.columns:
                continue
            if not np.issubdtype(df_clean[col].dtype, np.number):
                continue
            
            col_data = df_clean[col].astype(float)
            lower = np.percentile(col_data, lower_pct)
            upper = np.percentile(col_data, upper_pct)
            
            if lower < upper:
                # Normal case: percentile-based clipping
                df_clean[col] = col_data.clip(lower, upper)
            else:
                # Edge case: near-constant column with possible spikes
                # Use median-based fallback to catch artifacts
                median = np.median(col_data)
                if median != 0:
                    max_val = col_data.max()
                    min_val = col_data.min()
                    
                    # Detect spikes: values > spike_threshold times median
                    if max_val > abs(median) * spike_threshold or min_val < -abs(median) * spike_threshold:
                        clip_upper = abs(median) * spike_threshold
                        clip_lower = -abs(median) * spike_threshold if median < 0 else median / spike_threshold
                        df_clean[col] = col_data.clip(clip_lower, clip_upper)
    
    final_len = len(df_clean)
    total_dropped = initial_len - final_len
    
    if verbose and total_dropped > 0:
        pct = total_dropped / initial_len * 100
        logger.info(f"Sanitization: dropped {total_dropped} rows ({pct:.2f}%)")
        for reason, count in dropped_reasons.items():
            logger.info(f"  - {reason}: {count}")
    
    return df_clean


def validate_features(
    df: pd.DataFrame,
    feature_names: List[str],
    extreme_percentile: float = 99.9,
) -> Tuple[bool, dict]:
    """
    Validate DataFrame has clean features without modifying.
    
    Returns:
        (is_valid, stats_dict)
    """
    available = [f for f in feature_names if f in df.columns]
    
    if not available:
        return False, {"error": "no_features"}
    
    stats = {
        "total_rows": len(df),
        "features_found": len(available),
        "features_missing": len(feature_names) - len(available),
    }
    
    numeric_cols = df[available].select_dtypes(include=[np.number]).columns.tolist()
    
    # Check for inf
    inf_count = np.isinf(df[numeric_cols]).sum().sum() if numeric_cols else 0
    stats["inf_values"] = int(inf_count)
    
    # Check for nan
    nan_count = df[available].isna().sum().sum()
    stats["nan_values"] = int(nan_count)
    
    # Check for extreme values
    extreme_count = 0
    lower_pct = 100 - extreme_percentile
    upper_pct = extreme_percentile
    
    for col in numeric_cols:
        col_data = df[col].astype(float)
        lower = np.percentile(col_data, lower_pct)
        upper = np.percentile(col_data, upper_pct)
        extreme_count += ((col_data < lower) | (col_data > upper)).sum()
    
    stats["extreme_values"] = int(extreme_count)
    
    is_valid = (inf_count == 0) and (nan_count == 0)
    
    return is_valid, stats