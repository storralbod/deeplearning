import torch
import numpy as np
import matplotlib.pyplot as plt

def quantile_loss(y, y_hat, quantiles, weights=[0.2, 0.6, 0.2], dominant_period=24, reg_weight=0.1):
    """
    Compute weighted quantile loss with periodic regularization to enforce quantile ordering.
    Args:
        y (Tensor): Targets of shape [batch_size, forecast_horizon, 1].
        y_hat (Tensor): Predictions of shape [batch_size, len(quantiles), forecast_horizon].
        quantiles (list): List of quantiles.
        weights (list or None): Weights for each quantile. If None, uniform weighting is applied.
        dominant_period (int): The dominant period of the target in time steps.
        reg_weight (float): Regularization weight for periodic alignment.
    Returns:
        loss (Tensor): Scalar weighted quantile loss.
    """
    if weights is None:
        weights = [1.0] * len(quantiles)

    # Normalize weights to sum to 1 for stability
    weights = torch.tensor(weights, device=y.device, dtype=y.dtype)
    weights = weights / weights.sum()

    losses = []

    # Reshape targets to [batch_size, 1, forecast_horizon] for broadcasting
    y = y.permute(0, 2, 1)  # Change shape from [batch_size, forecast_horizon, 1] to [batch_size, 1, forecast_horizon]

    # Compute FFT frequencies for the periodic regularization
    forecast_horizon = y_hat.shape[-1]
    freq_axis = torch.fft.rfftfreq(forecast_horizon, d=1.0).to(y.device)  # Frequency axis
    target_freq_idx = (1.0 / dominant_period) / freq_axis[1]  # Convert period to frequency index

    for i, (q, w) in enumerate(zip(quantiles, weights)):
        # Align dimensions and calculate errors
        errors = y - y_hat[:, i, :].unsqueeze(1)  # [batch_size, 1, forecast_horizon]
        quantile_loss = torch.max(q * errors, (q - 1) * errors).mean()

        # Perform FFT once for this quantile prediction
        fft_result = torch.fft.rfft(y_hat[:, i, :], dim=-1)
        amplitudes = torch.abs(fft_result)  # Amplitudes of frequency components

        # Identify dominant frequency in the predictions
        dominant_idx = torch.argmax(amplitudes, dim=-1)
        
        # Align prediction dominant frequency with target dominant frequency
        reg_loss = reg_weight * torch.mean((dominant_idx - target_freq_idx) ** 2)

        # Add periodic regularization to quantile loss
        total_loss = quantile_loss + reg_loss
        losses.append(w * total_loss)

    return torch.sum(torch.stack(losses))


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
