import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import MinMaxScaler

# Import наших модулей
import sys
sys.path.append('scripts')
from time_embeddings import TimeEmbeddings, add_time_features
from triple_barrier_labels import create_triple_barrier_labels

class ImprovedCNN(nn.Module):
    def __init__(self, input_features, time_emb_dim=18):
        super().__init__()
        
        # Time embeddings
        self.time_emb = TimeEmbeddings()
        
        # CNN для price features
        total_features = input_features + 12
        self.conv1 = nn.Conv1d(total_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 3)  # 3 classes: UP, NEUTRAL, DOWN
        
    def forward(self, x, hour, dow, dom):
        # Time embeddings
        time_emb = self.time_emb(hour, dow, dom)  # (batch, 18)
        
        # Expand для каждого timestep
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, 60, 18)
        
        # Concatenate с price features
        x = torch.cat([x, time_emb_expanded], dim=2)  # (batch, 60, input_features+18)
        
        # CNN
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def prepare_data_with_triple_barrier():
    """
    Подготовка данных с triple-barrier labels и time features
    """
    print("\n" + "="*60)
    print("PREPARING DATA WITH TRIPLE-BARRIER LABELS")
    print("="*60)
    
    # Load BTC data
    df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
    print(f"Loaded {len(df)} bars")
    
    # Add time features
    df = add_time_features(df)
    
    # Create triple-barrier labels
    print("\nCreating triple-barrier labels...")
    labels = create_triple_barrier_labels(df, upper=0.02, lower=-0.01, horizon=96)
    df['label'] = labels
    
    # Remove None labels
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label in [-1, 0, 1]:
        count = label_counts.get(label, 0)
        pct = count / len(df) * 100
        name = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}[label]
        print(f"  {name} ({label}): {count} ({pct:.1f}%)")
    
    # Create features (simplified - using only price data for now)
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    X_data = df[feature_cols].values
    y_data = df['label'].values + 1  # Convert -1,0,1 to 0,1,2
    time_data = df[['hour', 'day_of_week', 'day_of_month']].values
    
    # Create sequences
    sequence_length = 60
    X_seq, y_seq, time_seq = [], [], []
    
    for i in range(len(X_data) - sequence_length):
        X_seq.append(X_data[i:i+sequence_length])
        y_seq.append(y_data[i+sequence_length])
        time_seq.append(time_data[i+sequence_length])
    
    X = np.array(X_seq)
    y = np.array(y_seq)
    time_features = np.array(time_seq)
    
    print(f"\nSequences created: {X.shape}")
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    time_train, time_test = time_features[:split_idx], time_features[split_idx:]
    
    # Scale
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # To tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Time features to tensors
    time_train = torch.tensor(time_train, dtype=torch.long)
    time_test = torch.tensor(time_test, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test, time_train, time_test

def train_and_evaluate():
    """
    Обучение модели с улучшениями
    """
    # Prepare data
    X_train, X_test, y_train, y_test, time_train, time_test = prepare_data_with_triple_barrier()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Model
    model = ImprovedCNN(input_features=5)
    model.to(device)
    
    # Loss (weighted для imbalanced classes)
    class_counts = np.bincount(y_train.numpy())
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # OneCycleLR
    train_dataset = TensorDataset(X_train, time_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=50,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    time_train = time_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    time_test = time_test.to(device)
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_time, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_time = batch_time.to(device)
            batch_y = batch_y.to(device)
            
            # Split time features
            hour = batch_time[:, 0]
            dow = batch_time[:, 1]
            dom = batch_time[:, 2]
            
            optimizer.zero_grad()
            outputs = model(batch_x, hour, dow, dom)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        hour_test = time_test[:, 0]
        dow_test = time_test[:, 1]
        dom_test = time_test[:, 2]
        
        outputs = model(X_test, hour_test, dow_test, dom_test)
        _, predicted = torch.max(outputs, 1)
        
        # Accuracy
        accuracy = (predicted == y_test).float().mean()
        
        # Per-class metrics
        y_np = y_test.cpu().numpy()
        pred_np = predicted.cpu().numpy()
        
        # Win rate (только для UP predictions)
        up_predictions = (pred_np == 2)  # Class 2 = UP
        if up_predictions.sum() > 0:
            up_correct = ((pred_np == 2) & (y_np == 2)).sum()
            win_rate = up_correct / up_predictions.sum()
        else:
            win_rate = 0
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_np, pred_np)
        
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"Win Rate (UP predictions): {win_rate*100:.2f}%")
        print(f"\nConfusion Matrix:")
        print("              Predicted")
        print("           DOWN  NEUTRAL  UP")
        for i, label in enumerate(['DOWN', 'NEUTRAL', 'UP']):
            print(f"Actual {label:7s} {cm[i]}")
        
        # Expected value estimation
        if up_predictions.sum() > 0:
            avg_win = 0.02  # 2% target
            avg_loss = 0.01  # 1% stop
            commission = 0.002
            
            exp_value = win_rate * (avg_win - commission) - (1 - win_rate) * (avg_loss + commission)
            
            print(f"\n💰 TRADING METRICS:")
            print(f"Win Rate: {win_rate*100:.1f}%")
            print(f"Expected Value: {exp_value*100:.3f}% per trade")
            
            if exp_value > 0:
                print("✅ POTENTIALLY PROFITABLE")
            else:
                print("❌ NOT PROFITABLE")
                print(f"Need Win Rate > {(avg_loss + commission)/(avg_win + avg_loss)*100:.1f}%")
        
        # Save
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/triple_barrier_v5.pth")
        
        metrics = {
            'accuracy': float(accuracy),
            'win_rate': float(win_rate),
            'expected_value': float(exp_value) if up_predictions.sum() > 0 else 0,
            'confusion_matrix': cm.tolist()
        }
        
        os.makedirs("reports", exist_ok=True)
        with open('reports/day5_v5_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Model saved: models/triple_barrier_v5.pth")
        print(f"📊 Metrics saved: reports/day5_v5_metrics.json")

if __name__ == "__main__":
    train_and_evaluate()