import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==================== STEP 1: Custom Dataset Class ====================
class DemandDataset(Dataset):
    """Custom PyTorch Dataset for demand forecasting"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== STEP 2: Data Preprocessing ====================
def create_sequences(data, seq_length=14, forecast_horizon=7):
    """
    Create sequences for time series forecasting
    Args:
        data: DataFrame with features
        seq_length: Number of past days to use
        forecast_horizon: Number of days to predict
    """
    sequences = []
    targets = []
    
    # Group by store and product
    for (store, product), group in data.groupby(['store_id', 'product_id']):
        group = group.sort_values('date').reset_index(drop=True)
        
        # Extract features
        features = group[['sales', 'price', 'promotion', 'inventory', 
                         'day_of_week', 'month', 'is_holiday']].values
        
        # Create sequences
        for i in range(len(features) - seq_length - forecast_horizon + 1):
            seq = features[i:i + seq_length]
            target = features[i + seq_length:i + seq_length + forecast_horizon, 0]  # sales only
            sequences.append(seq)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)


# ==================== STEP 3: Model Architecture ====================
class LSTMDemandForecaster(nn.Module):
    """
    LSTM-based demand forecasting model
    Architecture: LSTM → Dropout → Fully Connected Layers
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.2, forecast_horizon=7):
        super(LSTMDemandForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, forecast_horizon)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


# ==================== STEP 4: Training Loop ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=50, device='cpu'):
    """
    Training loop with validation
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


# ==================== STEP 5: Main Execution ====================
def main():
    # Load data
    df = pd.read_csv('retail_sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(df, seq_length=14, forecast_horizon=7)
    
    # Normalize features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = DemandDataset(X_train, y_train)
    test_dataset = DemandDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = LSTMDemandForecaster(
        input_size=7,  # 7 features
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        forecast_horizon=7
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, 
        optimizer, num_epochs=50, device=device
    )
    
    # Evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            # Denormalize predictions
            outputs_denorm = scaler_y.inverse_transform(outputs.cpu().numpy())
            targets_denorm = scaler_y.inverse_transform(targets.numpy())
            
            predictions.extend(outputs_denorm)
            actuals.extend(targets_denorm)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    print(f"\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save model for deployment
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, 'demand_forecaster_pytorch.pth')
    
    print("\nModel saved successfully!")


if __name__ == "__main__":
    main()
