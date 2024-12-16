import torch
import numpy as np
import matplotlib.pyplot as plt

def quantile_loss(y, y_hat, quantiles):
    """
    Compute quantile loss and enforce quantile ordering.
    Args:
        y (Tensor): Targets of shape [batch_size, forecast_horizon, 1].
        y_hat (Tensor): Predictions of shape [batch_size, len(quantiles), forecast_horizon].
        quantiles (list): List of quantiles.
    Returns:
        loss (Tensor): Scalar quantile loss.
    """
    losses = []

    # Reshape targets to [batch_size, 1, forecast_horizon] for broadcasting
    y = y.permute(0, 2, 1)  # Change shape from [batch_size, forecast_horizon, 1] to [batch_size, 1, forecast_horizon]

    for i, q in enumerate(quantiles):
        # Align dimensions and calculate errors
        errors = y - y_hat[:, i, :].unsqueeze(1)  # [batch_size, 1, forecast_horizon]
        losses.append(torch.max(q * errors, (q - 1) * errors).mean())

    # Enforce quantile ordering: Q[i] <= Q[i+1]
    ordering_penalty = 0
    for i in range(len(quantiles) - 1):
        ordering_penalty += torch.mean(torch.relu(y_hat[:, i, :] - y_hat[:, i + 1, :]))

    return torch.mean(torch.stack(losses)) + 0.5 * ordering_penalty



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
                dense_inputs, spaced_inputs, future_inputs, targets = batch
                dense_inputs, spaced_inputs, future_inputs = (
                    dense_inputs.to(device),
                    spaced_inputs.to(device),
                    future_inputs.to(device),
                )
                preds = model(dense_inputs, spaced_inputs, future_inputs)
            else:
                dense_inputs, future_inputs, targets = batch
                dense_inputs, future_inputs = dense_inputs.to(device), future_inputs.to(device)
                preds = model(dense_inputs, future_inputs)

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

# Plot Forecasts
def plot_forecasts(forecasts, true_values, sample_idx, quantiles, forecast_horizon):
    """
    Visualizes the forecasts vs. true values for a single sample.
    """
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
    plt.title("Quantile Forecasts vs True Values")
    plt.xlabel("Time Step")
    plt.ylabel("DA Value")
    plt.legend()
    plt.grid(True)
    plt.show()
