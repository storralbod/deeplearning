import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train LSTM model for price prediction.")
parser.add_argument('--data', type=str, required=True, help="Path to the input data file (CSV).")
args = parser.parse_args()

# Set the data path from the command line
file_path = args.data

# Ensure the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found: {file_path}")

print(f"Using data file: {file_path}")

# Define input and output sequence lengths
sequence_length = 336  # Use 2 weeks of history
output_sequence_length = 24  # Predict the next 24 hours
batch_size = 32

# Prepare data
train_data = []
test_data = []

# Initialize scalers
scaler_price1 = MinMaxScaler(feature_range=(0, 1))
scaler_price2 = MinMaxScaler(feature_range=(0, 1))
scaler_features = MinMaxScaler(feature_range=(0, 1))

# First pass: Fit the scalers
chunksize = 10_000
reader = pd.read_csv(file_path, chunksize=chunksize)

for chunk in reader:
    price1 = chunk[['Price1']].values
    price2 = chunk[['Price2']].values
    features = chunk[['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos']].values
    scaler_price1.partial_fit(price1)
    scaler_price2.partial_fit(price2)
    scaler_features.partial_fit(features)

# Second pass: Transform the data
reader = pd.read_csv(file_path, chunksize=chunksize)

for chunk in reader:
    price1_scaled = scaler_price1.transform(chunk[['Price1']].values)
    price2_scaled = scaler_price2.transform(chunk[['Price2']].values)
    features_scaled = scaler_features.transform(chunk[['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos']].values)

    # Combine scaled features
    combined_features = np.hstack([price1_scaled, price2_scaled, features_scaled])

    if len(combined_features) < sequence_length + output_sequence_length:
        continue  # Skip chunks that are too small for the sequence lengths

    X, y = [], []
    for i in range(len(combined_features) - sequence_length - output_sequence_length):
        X.append(combined_features[i:i + sequence_length])  # Input sequence
        y.append(price1_scaled[i + sequence_length:i + sequence_length + output_sequence_length, 0])  # Target sequence

    train_data.append((X[:-5], y[:-5]))  # All but the last 5 entries as training
    test_data = (X[-5:], y[-5:])         # Last 5 entries as test data

# Combine training data
X_train = np.concatenate([data[0] for data in train_data], axis=0)
y_train = np.concatenate([data[1] for data in train_data], axis=0)

# Prepare test data
X_test, y_test = np.array(test_data[0]), np.array(test_data[1])

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

# Training loop
epochs = 50
model_save_path = './deeplearn/model/lstm_model.pth'

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.6f}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# Model evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).numpy()

# Save predictions and actual values for later analysis
predictions_save_path = './deeplearn/model/predictions.npy'
actual_save_path = './deeplearn/model/actual.npy'
np.save(predictions_save_path, y_test_pred)
np.save(actual_save_path, y_test.numpy())
print(f"Predictions saved at {predictions_save_path}")
print(f"Actual values saved at {actual_save_path}")
