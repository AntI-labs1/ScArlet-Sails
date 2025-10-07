import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60, target_column="target_binary"):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)
    
    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    
    # Target ПЕРВЫМ
    FUTURE_BARS = 4
    THRESHOLD = 0.002
    combined_df['target_binary'] = (combined_df['close'].shift(-FUTURE_BARS) > combined_df['close'] * (1 + THRESHOLD)).astype(int)
    combined_df = combined_df[:-FUTURE_BARS].copy()
    
    # ТОЛЬКО базовые features - БЕЗ индикаторов
    feature_cols = ['open', 'high', 'low', 'volume']  # БЕЗ close и БЕЗ индикаторов
    
    # Удалить NaN
    combined_df = combined_df.dropna()
    
    X_data = combined_df[feature_cols].values
    y_data = combined_df['target_binary'].values
    
    scaler_X = MinMaxScaler()
    scaled_X_data = scaler_X.fit_transform(X_data)
    
    X, y = [], []
    for i in range(len(scaled_X_data) - sequence_length):
        X.append(scaled_X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    return X, y, scaler_X, None

if __name__ == "__main__":
    X, y, _, _ = load_and_preprocess_data()
    print(f"Minimal features X: {X.shape}")
    print(f"Features: {X.shape[2]}")