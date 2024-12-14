import torch
import copy
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_prep import create_datasets, Dataset

def prepare_data(file_path, lags, temporal_cols, gas_col, da_col, id_col, forecast_horizon):
    # Load data
    data = pd.read_csv(file_path)
    data = data.dropna().reset_index(drop=True)
    data = data.drop(columns=['Volume_MWh', 'Diff', 'Year', 'Month', 'Day', 'Hour'])
    output_data = np.array(data.loc[:, da_col]).reshape(-1, 1)
    data = np.array(data)
    
    train_data, test_data, val_data = create_datasets(data, output_data, 2, forecast_horizon, Dataset, p_train=0.8, p_val=0, p_test=0.2)

    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    val_data = pd.DataFrame(val_data)

    # Scaling
    scale_cols = [da_col, id_col, gas_col]
    scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in scale_cols}

    for col in scale_cols:
        train_data[col] = scalers[col].fit_transform(train_data[[col]])
        test_data[col] = scalers[col].transform(test_data[[col]])

    return train_data, test_data, scalers

def prepare_data_with_weather(file_path, historical_columns, predictive_columns, temporal_cols, target_column, sequence_length, forecast_horizon, test_split_ratio=0.1):
    # Load the data
    data = pd.read_csv(file_path)

    # Convert to datetime for filtering
    data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    start_date = "2023-01-01"
    end_date = "2024-06-30"
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data = data.drop(columns=['Date'])  # Drop 'Date' column if not needed

    # Shift the predictive columns to align with the target
    data[predictive_columns] = data[predictive_columns].shift(-forecast_horizon)

    # Drop rows with missing values
    data = data.dropna(subset=historical_columns + predictive_columns + [target_column])

    # Scaling
    scaler_historical = MinMaxScaler(feature_range=(0, 1))
    scaler_predictive = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    # Scale features and target
    historical_scaled = scaler_historical.fit_transform(data[historical_columns])
    predictive_scaled = scaler_predictive.fit_transform(data[predictive_columns])
    target_scaled = scaler_target.fit_transform(data[[target_column]])

    # Concatenate scaled historical and temporal features
    features_scaled = np.hstack([historical_scaled, data[temporal_cols].values])

    # Prepare sequences
    x_past, x_future, y = [], [], []
    for i in range(len(features_scaled) - sequence_length - forecast_horizon):
        # Historical features for x_past
        x_past.append(features_scaled[i:i + sequence_length])

        # Predictive + temporal features for x_future
        future_features = np.hstack([predictive_scaled[i + sequence_length:i + sequence_length + forecast_horizon],
                                     data[temporal_cols].values[i + sequence_length:i + sequence_length + forecast_horizon]])
        x_future.append(future_features)

        # Target values
        y.append(target_scaled[i + sequence_length:i + sequence_length + forecast_horizon].flatten())

    # Convert to numpy arrays
    x_past = np.array(x_past)
    x_future = np.array(x_future)
    y = np.array(y)

    # Train-test split
    split_idx = int(len(x_past) * (1 - test_split_ratio))
    x_past_train, x_past_test = x_past[:split_idx], x_past[split_idx:]
    x_future_train, x_future_test = x_future[:split_idx], x_future[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to PyTorch tensors
    x_past_train = torch.tensor(x_past_train, dtype=torch.float32)
    x_future_train = torch.tensor(x_future_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_past_test = torch.tensor(x_past_test, dtype=torch.float32)
    x_future_test = torch.tensor(x_future_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(x_past_train, x_future_train, y_train)
    test_dataset = TensorDataset(x_past_test, x_future_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training data shape: x_past: {x_past_train.shape}, x_future: {x_future_train.shape}, y: {y_train.shape}")
    print(f"Testing data shape: x_past: {x_past_test.shape}, x_future: {x_future_test.shape}, y: {y_test.shape}")

    input_size = x_past_train.shape[2]  # Size of x_past features

    return train_loader, test_loader, scaler_historical, scaler_predictive, scaler_target, input_size



# Quantile Loss Function
def quantile_loss(y, y_hat, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = y.unsqueeze(1) - y_hat[:, i, :]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())

    # Enforce quantile ordering: Q[i] <= Q[i+1]
    ordering_penalty = 0
    for i in range(len(quantiles) - 1):
        ordering_penalty += torch.mean(torch.relu(y_hat[:, i, :] - y_hat[:, i + 1, :]))  # Penalize inversions

    return torch.mean(torch.stack(losses)) + 0.5 * ordering_penalty

def quantile_loss_simple(y_true, y_pred, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true.unsqueeze(1) - y_pred[:, i, :]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())
    return torch.mean(torch.stack(losses))

def train_and_evaluate_model(model, train_loader, val_loader, optimizer, quantiles, device, num_epochs=3, clip_grad=5.0):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # Training phase
        for x_past, x_future, y in train_loader:
            x_past, x_future, y = x_past.to(device), x_future.to(device), y.to(device)

            optimizer.zero_grad()

            # Predictions
            forecasts = model(x_past, x_future)

            # Loss calculation
            loss = quantile_loss(y, forecasts, quantiles)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_past, x_future, y in val_loader:
                x_past, x_future, y = x_past.to(device), x_future.to(device), y.to(device)
                forecasts = model(x_past, x_future)

                loss = quantile_loss(y, forecasts, quantiles)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Save the best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    plot_training_validation_loss(train_losses, val_losses)

    return model, train_losses, val_losses


def predict_model(model, test_loader, scaler_target, da_col, quantiles, forecast_horizon, device):
    """
    Predicts using the trained model and scales back the predictions.
    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test set.
        scaler_target: MinMaxScaler object for the target variable.
        da_col: Column name of the target variable.
        quantiles: List of quantiles used during training.
        forecast_horizon: Number of time steps to forecast.
        device: Device (CPU/GPU) for computation.
    Returns:
        forecast_inv: Inverse-scaled predictions.
        true_inv: Inverse-scaled true values.
    """
    model.eval()
    forecasts, true_values = [], []

    with torch.no_grad():
        for x_past, x_future, y in test_loader:
            x_past, x_future = x_past.to(device), x_future.to(device)
            preds = model(x_past, x_future)  # Model output
            print(f"Model output shape: {preds.shape}")  # Debug the output shape
            forecasts.append(preds.cpu())
            true_values.append(y)

    forecasts = torch.cat(forecasts, dim=0).numpy()  # Combine batches
    true_values = torch.cat(true_values, dim=0).numpy()  # Combine batches

    print(f"Combined forecasts shape: {forecasts.shape}")
    print(f"True values shape: {true_values.shape}")

    # Check if forecasts match the expected shape
    if forecasts.shape[-1] != forecast_horizon or forecasts.shape[1] != len(quantiles):
        raise ValueError(
            f"Forecasts shape mismatch! Expected [batch_size, {len(quantiles)}, {forecast_horizon}], "
            f"but got {forecasts.shape}."
        )

    # Inverse scaling
    forecast_inv = []
    for q in range(len(quantiles)):
        scaled = forecasts[:, q, :].reshape(-1, 1)  # Reshape for scaler
        print(f"Quantile {q}, Scaled shape before inverse_transform: {scaled.shape}")
        inv_scaled = scaler_target.inverse_transform(scaled).reshape(-1, forecast_horizon)  # Reshape back
        print(f"Quantile {q}, Inverse transformed shape: {inv_scaled.shape}")
        forecast_inv.append(inv_scaled)

    forecast_inv = np.stack(forecast_inv, axis=1)  # [batch_size, len(quantiles), forecast_horizon]
    true_inv = scaler_target.inverse_transform(true_values.reshape(-1, 1)).reshape(-1, forecast_horizon)  # [batch_size, forecast_horizon]

    return forecast_inv, true_inv


def plot_forecasts(forecasts, true_values, sample_idx, quantiles, forecast_horizon):

    q_10, q_50, q_90 = forecasts[sample_idx, 0, :], forecasts[sample_idx, 1, :], forecasts[sample_idx, 2, :]
    true_vals = true_values[sample_idx, :]
    time_steps = np.arange(forecast_horizon)

    plt.figure(figsize=(12, 6))
    plt.fill_between(time_steps, q_10, q_90, color='gray', alpha=0.3, label=f"{quantiles[0]} - {quantiles[2]} Quantile Range")
    plt.plot(time_steps, q_50, label=f"{quantiles[1]} Quantile (Median)", color='blue', linewidth=2)
    plt.plot(time_steps, true_vals, label="True Values", color='black', linestyle='--', linewidth=2)
    plt.title("Quantile Forecasts vs True Values")
    plt.xlabel("Time Step")
    plt.ylabel("DA Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_validation_loss(train_losses, val_losses):
    """
    Plots training and validation losses.
    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
    """
    epochs = range(1, len(train_losses) + 1)  # Create x-axis with epoch numbers

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
# plot_training_validation_loss(tr_loss, val_loss)
