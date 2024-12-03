import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# File paths
file_path = '../TrainingData/trainingdata_23_24.csv'

# Define input and output sequence lengths
sequence_length = 336  # Use 2 weeks of history
output_sequence_length = 24  # Predict the next 24 hours
batch_size = 32

# Load data in chunks and split into training and test datasets
train_data = []
test_data = []
test_split_ratio = 0.1  # Use the last 10% for testing

# Read the file in chunks
chunksize = 10_000
reader = pd.read_csv(file_path, chunksize=chunksize)

for chunk in reader:
    # Prepare input features and target prices
    price1 = chunk[['Price1']].values
    price2 = chunk[['Price2']].values
    features = chunk[['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos']].values

    combined_features = np.hstack([price1, price2, features])

    if len(combined_features) < sequence_length + output_sequence_length:
        continue  # Skip chunks that are too small for the sequence lengths

    X, y = [], []
    for i in range(len(combined_features) - sequence_length - output_sequence_length):
        X.append(combined_features[i:i + sequence_length])  # Input sequence
        y.append(price1[i + sequence_length:i + sequence_length + output_sequence_length, 0])  # Target sequence
    
    if len(test_data) == 0:  # Reserve the last portion of the data for testing
        test_start_idx = int(len(X) * (1 - test_split_ratio))
        train_data.append((X[:test_start_idx], y[:test_start_idx]))
        test_data = (X[test_start_idx:], y[test_start_idx:])
    else:
        train_data.append((X, y))

# Combine training data
X_train = np.concatenate([data[0] for data in train_data], axis=0)
y_train = np.concatenate([data[1] for data in train_data], axis=0)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

# Prepare test data
X_test, y_test = np.array(test_data[0]), np.array(test_data[1])

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Model parameters
input_size = X_train.shape[2]
hidden_size = 50
output_size = output_sequence_length

model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Checkpoint handling
start_epoch = 0
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint.pth")

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    print(f"Resumed training from epoch {start_epoch}.")
else:
    train_losses = []
    val_losses = []

# Training loop with validation
epochs = 50

for epoch in range(start_epoch, start_epoch + epochs):
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Record training loss
    train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluate on validation set
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_val_loss += loss.item()
    
    val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{start_epoch + epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, checkpoint_path)

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid()
plt.show()

# Model evaluation and visualization
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).numpy()

# Plot predictions vs actual for test data
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test[0])), y_test[0], label="Actual Prices", color="blue")
plt.plot(range(len(y_test_pred[0])), y_test_pred[0], label="Predicted Prices", color="red")
plt.title("Predicted vs Actual Prices for the Test Set")
plt.xlabel("Hour")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()
