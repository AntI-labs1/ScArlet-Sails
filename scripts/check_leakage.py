import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from prepare_data import load_and_preprocess_data

# Загрузить чистые данные
df = pd.read_parquet('data/raw/BTC_USDT_15m.parquet')

# Создать простой target БЕЗ индикаторов
df['simple_target'] = (df['close'].shift(-4) > df['close'] * 1.002).astype(int)
df = df[:-4]

print("=== CORRELATION ANALYSIS ===")

# Корреляция базовых features с target
basic_features = ['open', 'high', 'low', 'close', 'volume']
for feat in basic_features:
    corr = df[feat].corr(df['simple_target'])
    print(f"{feat} correlation with target: {corr:.4f}")

print("\nIf close correlation > 0.1 → potential leakage")