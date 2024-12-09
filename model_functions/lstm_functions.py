import torch
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path, lags, temporal_cols, gas_col, da_col, id_col, forecast_horizon):
    # Load data
    data = pd.read_csv(file_path)
    data = data.dropna().reset_index(drop=True)
    data = data.drop(columns=['Volume_MWh', 'Diff', 'Year', 'Month', 'Day', 'Hour'])
    # # Create lagged features
    # for col in [da_col, id_col]:
    #     for lag in lags:
    #         data[f'{col}_t-{lag}'] = data[col].shift(lag)

    # for lag in [24, 48]:
    #     data[f'Price_EUR_MWh_t-{lag}'] = data['Price_EUR_MWh'].shift(lag)

    # Drop rows with NaN values
    # data = data.dropna().reset_index(drop=True)

    # Train-test split
    train_size = max(lags)+forecast_horizon+1
    train_data = data.iloc[:train_size].reset_index(drop=True)
    test_data = data.iloc[train_size:].reset_index(drop=True)

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

# Quantile Diversity Regularization
def quantile_diversity_loss(y_hat, quantiles):
    diversity_penalty = 0
    for i in range(len(quantiles) - 1):
        diversity_penalty += torch.abs(y_hat[:, i, :] - y_hat[:, i + 1, :]).mean()
    return 0.01 * diversity_penalty

def temporal_smoothing_loss(y_hat):
    return torch.mean(torch.abs(y_hat[:, :, 1:] - y_hat[:, :, :-1]))

def train_model(model, train_loader, optimizer, quantiles, device, num_epochs=3, clip_grad=5.0):
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for x_past, x_future, y in train_loader:
            x_past, x_future, y = x_past.to(device), x_future.to(device), y.to(device)

            optimizer.zero_grad()

            # Predictions
            forecasts = model(x_past)

            # Loss calculation
            loss = quantile_loss(y, forecasts, quantiles) + 0.01 * quantile_diversity_loss(forecasts, quantiles) + 0.01 * temporal_smoothing_loss(forecasts)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

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
