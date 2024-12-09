import torch
import copy
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
            forecasts = model(x_past)

            # Loss calculation
            loss = quantile_loss(y, forecasts, quantiles)
            train_losses.append(loss.item())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_past, x_future, y in val_loader:
                x_past, x_future, y = x_past.to(device), x_future.to(device), y.to(device)
                forecasts = model(x_past)

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
    # Return both the model and the loss history for analysis
    return model, train_losses, val_losses

def predict_model(model, test_loader, scalers, da_col, quantiles, forecast_horizon, device):
    model.eval()
    forecasts, true_values = [], []

    with torch.no_grad():
        for x_past, x_future, y in test_loader:
            x_past, x_future = x_past.to(device), x_future.to(device)
            preds = model(x_past, x_future)
            forecasts.append(preds.cpu())
            true_values.append(y)

    forecasts = torch.cat(forecasts, dim=0).numpy()
    true_values = torch.cat(true_values, dim=0).numpy()
    # Inverse scaling
    forecast_inv = []
    for q in range(len(quantiles)):
        scaled = forecasts[:, q, :].reshape(-1, 1)
        forecast_inv.append(scalers[da_col].inverse_transform(scaled).reshape(-1, forecast_horizon))
    forecast_inv = np.stack(forecast_inv, axis=1)

    true_inv = scalers[da_col].inverse_transform(true_values.reshape(-1, 1)).reshape(-1, forecast_horizon)

    # Check quantile consistency
    for sample in range(forecast_inv.shape[0]):
        for t in range(forecast_inv.shape[2]):
            for i in range(len(quantiles) - 1):
                if forecast_inv[sample, i, t] > forecast_inv[sample, i + 1, t]:
                    print(f"Quantile inconsistency at sample {sample}, time {t}: Q{quantiles[i]} > Q{quantiles[i+1]}")

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
    - train_losses (list): List of training loss values.
    - val_losses (list): List of validation loss values.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
# plot_training_validation_loss(tr_loss, val_loss)
