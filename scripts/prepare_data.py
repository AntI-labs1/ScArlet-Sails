import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No 15m parquet files found.")
    
    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    
    # Target FIRST
    FUTURE_BARS = 4
    THRESHOLD = 0.002
    combined_df['target_binary'] = (combined_df['close'].shift(-FUTURE_BARS) > combined_df['close'] * (1 + THRESHOLD)).astype(int)
    combined_df = combined_df[:-FUTURE_BARS].copy()
    
    # Indicators
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    combined_df['RSI_14'] = calculate_rsi(combined_df['close'], 14)
    combined_df['EMA_9'] = combined_df['close'].ewm(span=9, adjust=False).mean()
    combined_df['EMA_21'] = combined_df['close'].ewm(span=21, adjust=False).mean()
    combined_df['BB_middle'] = combined_df['close'].rolling(window=20).mean()
    combined_df['BB_std'] = combined_df['close'].rolling(window=20).std()
    combined_df['BB_upper'] = combined_df['BB_middle'] + (combined_df['BB_std'] * 2)
    combined_df['BB_lower'] = combined_df['BB_middle'] - (combined_df['BB_std'] * 2)
    
    high_low = combined_df['high'] - combined_df['low']
    high_close = abs(combined_df['high'] - combined_df['close'].shift())
    low_close = abs(combined_df['low'] - combined_df['close'].shift())
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    combined_df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Remove close to avoid leakage
    combined_df = combined_df.dropna()
    feature_cols = [col for col in combined_df.columns if col not in ['target_binary', 'close']]
    
    X_data = combined_df[feature_cols].values
    y_data = combined_df['target_binary'].values
    
    # Create sequences WITHOUT scaling
    X_sequences = []
    y_targets = []
    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:i+sequence_length])
        y_targets.append(y_data[i+sequence_length])
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    # TEMPORAL SPLIT FIRST
    split_idx = int(len(X) * 0.8)
    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # FIT SCALER ONLY ON TRAIN
    scaler = MinMaxScaler()
    X_train_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    scaler.fit(X_train_reshaped)
    
    # TRANSFORM both using train parameters
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train_raw.shape)
    X_test_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test_raw.shape)
    
    # To tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_X.pkl")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Train X: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test X: {X_test.shape}, Test y: {y_test.shape}")
    print(f"Features: {X_train.shape[2]}")
    print("Data preparation complete - CORRECT VERSION")