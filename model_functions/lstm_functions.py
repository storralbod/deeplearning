import torch
import numpy as np
import matplotlib.pyplot as plt

def quantile_loss_with_rmse(y, y_hat, quantiles, dense_lookback, reg_weight=0.1, rmse_weight=1.0, lambda_range=0.1):
    """
    Compute quantile loss with RMSE for the middle quantile and periodic regularization.
    Args:
        y (Tensor): Targets of shape [batch_size, forecast_horizon, 1].
        y_hat (Tensor): Predictions of shape [batch_size, len(quantiles), forecast_horizon].
        quantiles (list): List of quantiles.
        dense_lookback (int): Length of dense lookback (for periodic regularization).
        reg_weight (float): Weight for periodic regularization.
        rmse_weight (float): Weight for RMSE term.
    Returns:
        total_loss (Tensor): Combined loss (scalar).
        rmse_middle (Tensor): RMSE for the middle quantile (scalar, for logging).
    """
    losses = []
    rmse_middle = None

    # Reshape targets to [batch_size, 1, forecast_horizon] for broadcasting
    y = y.permute(0, 2, 1)  # Change shape from [batch_size, forecast_horizon, 1] to [batch_size, 1, forecast_horizon]

    # Compute FFT frequencies for periodic regularization
    freq_axis = torch.fft.rfftfreq(dense_lookback, d=1.0).to(y.device)
    target_freq_idx = (1.0 / 24) / freq_axis[1]  # Target daily frequency

    for i, q in enumerate(quantiles):
        # Compute quantile loss
        errors = y - y_hat[:, i, :].unsqueeze(1)
        if q == 0.5:
            # Use RMSE for the middle quantile
            rmse_middle = torch.sqrt(torch.mean((y.squeeze(1) - y_hat[:, i, :]) ** 2))
            quantile_loss = rmse_middle * rmse_weight
        else:
            # Standard quantile loss for other quantiles
            quantile_loss = torch.max(q * errors, (q - 1) * errors).mean()

        # Apply periodic regularization for dense lookback (optional)
        if q == 0.5:  # Focus periodic regularization on the middle quantile
            fft_result = torch.fft.rfft(y_hat[:, i, :dense_lookback], dim=-1)  # Dense lookback portion
            amplitudes = torch.abs(fft_result)
            dominant_idx = amplitudes.argmax(dim=-1)
            reg_loss = reg_weight * torch.mean((dominant_idx - target_freq_idx) ** 2)
            quantile_loss += reg_loss

        losses.append(quantile_loss)

    quantile_range = y_hat[:, 2, :] - y_hat[:, 0, :]
    target_range = y.max(dim=2).values - y.min(dim=2).values  # True range
    range_penalty = lambda_range * ((quantile_range - target_range) ** 2).mean()

    total_loss = torch.sum(torch.stack(losses)) + range_penalty
    #total_loss = torch.sum(torch.stack(losses))
    return total_loss


def predict_model(model, test_loader, target_scaler, quantiles, forecast_horizon, device, spaced=False):
    """
    Predict using the model on the test dataset.
    
    Args:
        model: Trained model for prediction.
        test_loader: DataLoader containing the test dataset.
        target_scaler: Scaler used to scale the target values (output_data).
        quantiles: List of quantiles for the predictions.
        forecast_horizon: Number of timesteps in the forecast horizon.
        device: Torch device to use for computation (CPU/GPU).
        spaced (bool): Whether the model is a spaced model (True) or simple model (False).
    
    Returns:
        forecast_inv: Inverse scaled forecasts with shape [num_samples, len(quantiles), forecast_horizon].
        true_inv: Inverse scaled true values with shape [num_samples, forecast_horizon].
    """
    model.eval()
    forecasts, true_values = [], []

    with torch.no_grad():
        for batch in test_loader:
            # Separate inputs based on model type
            if spaced:
                dense_inputs, spaced_inputs, future_inputs, pca_inputs, targets = batch
                dense_inputs, spaced_inputs, future_inputs, pca_inputs = (
                    dense_inputs.to(device),
                    spaced_inputs.to(device),
                    future_inputs.to(device),
                    pca_inputs.to(device),
                )
                preds = model(dense_inputs, spaced_inputs, future_inputs, pca_inputs)
            else:
                dense_inputs, future_inputs, pca_inputs, targets = batch
                dense_inputs, future_inputs, pca_inputs = dense_inputs.to(device), future_inputs.to(device), pca_inputs.to(device)
                preds = model(dense_inputs, future_inputs, pca_inputs)

            forecasts.append(preds.cpu())
            true_values.append(targets)

    # Convert forecasts and true_values to NumPy arrays
    forecasts = torch.cat(forecasts, dim=0).numpy()  # Shape: [num_samples, len(quantiles), forecast_horizon]
    true_values = torch.cat(true_values, dim=0).numpy()  # Shape: [num_samples, forecast_horizon, 1]

    # Reshape true_values for inverse scaling
    true_values_reshaped = true_values.reshape(-1, 1)  # Flatten for inverse scaling
    true_inv = target_scaler.inverse_transform(true_values_reshaped).reshape(-1, forecast_horizon)

    # Reshape forecasts for inverse scaling
    forecast_inv = []
    for q in range(len(quantiles)):
        scaled_forecast = forecasts[:, q, :].reshape(-1, 1)  # Flatten for inverse scaling
        forecast_inv.append(target_scaler.inverse_transform(scaled_forecast).reshape(-1, forecast_horizon))

    forecast_inv = np.stack(forecast_inv, axis=1)  # Combine quantiles into a single array

    return forecast_inv, true_inv

def plot_forecasts(forecasts, true_values, quantiles, forecast_horizon):
    """
    Visualizes the best and worst forecasts vs. true values based on RMSE.
    Args:
        forecasts (ndarray): Predicted values of shape [num_samples, num_quantiles, forecast_horizon].
        true_values (ndarray): True values of shape [num_samples, forecast_horizon].
        quantiles (list): List of quantiles corresponding to forecasts.
        forecast_horizon (int): Number of timesteps in the forecast horizon.
    """
    # Compute RMSE for each sample
    rmse = np.sqrt(np.mean((forecasts[:, 1, :] - true_values) ** 2, axis=1))  # Median forecast (quantile[1])
    
    # Identify best and worst samples
    best_idx = np.argmin(rmse)
    worst_idx = np.argmax(rmse)
    
    def plot_single_forecast(sample_idx, title):
        q_10, q_50, q_90 = forecasts[sample_idx, 0, :], forecasts[sample_idx, 1, :], forecasts[sample_idx, 2, :]
        true_vals = true_values[sample_idx, :]
        time_steps = np.arange(forecast_horizon)
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(
            time_steps,
            q_10,
            q_90,
            color="gray",
            alpha=0.3,
            label=f"{quantiles[0]} - {quantiles[2]} Quantile Range",
        )
        plt.plot(time_steps, q_50, label=f"{quantiles[1]} Quantile (Median)", color="blue", linewidth=2)
        plt.plot(time_steps, true_vals, label="True Values", color="black", linestyle="--", linewidth=2)
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("DA Value")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plot best prediction
    plot_single_forecast(best_idx, f"Best Prediction (RMSE: {rmse[best_idx]:.4f})")
    
    # Plot worst prediction
    plot_single_forecast(worst_idx, f"Worst Prediction (RMSE: {rmse[worst_idx]:.4f})")

    # Print avergae RMSE
    print(f"Average RMSE: {np.mean(rmse):.4f}")
