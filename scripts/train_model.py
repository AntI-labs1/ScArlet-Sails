import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

class CNN1D(nn.Module):
    def __init__(self, input_features, sequence_length=60):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15, 64)  # 60->30->15 после 2 pooling
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, timesteps, features)
        x = x.transpose(1, 2)  # -> (batch, features, timesteps)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    import sys
    sys.path.append('scripts')
    from prepare_data import load_and_preprocess_data

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

    input_features = X_train.shape[2]
    model = CNN1D(input_features, sequence_length=60)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    print(f"Using device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {input_features}")

    # Training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("Training with CORRECTED architecture...")
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # НЕТ permute здесь - он в forward()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/30], Loss: {avg_loss:.4f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)  # НЕТ permute
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions.squeeze() == y_test).float().mean()
        
        # Additional metrics
        tp = ((predictions.squeeze() == 1) & (y_test == 1)).sum().item()
        fp = ((predictions.squeeze() == 1) & (y_test == 0)).sum().item()
        tn = ((predictions.squeeze() == 0) & (y_test == 0)).sum().item()
        fn = ((predictions.squeeze() == 0) & (y_test == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"True Positives: {tp}, False Positives: {fp}")
        print(f"True Negatives: {tn}, False Negatives: {fn}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/corrected_cnn_model.pth")
    print("\nModel saved. Training complete.")