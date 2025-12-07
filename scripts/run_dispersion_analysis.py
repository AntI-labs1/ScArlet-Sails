"""
Dispersion Analysis: сравнение P_rb vs P_ml на одинаковых данных.
"""
import pandas as pd
import numpy as np
from scipy import stats


def main():
    # 1. Load Model 2 predictions
    # (нужно модифицировать train_xgboost_v3.py чтобы сохранял proba)
    
    # 2. Load Model 1 predictions
    
    # 3. Align by timestamp
    
    # 4. Calculate dispersion metrics
    
    # ANOVA
    f_stat, p_value = stats.f_oneway(P_rb, P_ml)
    
    # Correlation
    corr = np.corrcoef(P_rb, P_ml)[0, 1]
    
    # Agreement rate (оба > 0.5 или оба < 0.5)
    agreement = ((P_rb > 0.5) == (P_ml > 0.5)).mean()
    
    # Output
    print(f"ANOVA F-stat: {f_stat:.2f}, p-value: {p_value:.6f}")
    print(f"Correlation: {corr:.4f}")
    print(f"Agreement rate: {agreement*100:.1f}%")
    
    # Когда модели согласны vs расходятся
    # ...