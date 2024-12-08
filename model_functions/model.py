import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class LSTM_multivariate_input_multi_step_forecaster(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,dropout, past_horizon, forecast_horizon):
        super().__init__()
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=dropout,batch_first=True)
        #self.lstm_autoregressive = nn.LSTM(1,hidden_size,num_layers,dropout=dropout,batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size,48*3),
            nn.ReLU(),
            nn.LayerNorm(48*3),
            nn.Dropout(dropout),
            nn.Linear(48*3,48*2),
            nn.ReLU(),
            nn.LayerNorm(48*2),
            nn.Dropout(dropout),
            nn.Linear(48*2,48*2),
            nn.ReLU(),
            nn.LayerNorm(48*2),
            nn.Dropout(dropout),
            nn.Linear(48*2,forecast_horizon)
        )

    def forward(self,x):

        if torch.isnan(x).any():
          nans = torch.isnan(x).any().sum()
          raise ValueError(f"Input contains {nans} NaN values")

        lstm_out, _ = self.lstm(x)
        forecast = self.mlp(lstm_out[:,-1,:])

        return forecast.unsqueeze(-1)
    
    import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LaggedPriceDataset(Dataset):
    """
    Custom Dataset for DA price prediction based on specific lagged inputs.
    """
    def __init__(self, data, lags, forecast_horizon):
        """
        Parameters:
        - data: DataFrame containing the input data
        - lags: List of lagged time steps to use as features
        - forecast_horizon: Number of future time steps to predict
        """
        self.data = data
        self.lags = lags
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        # Ensure we have enough data for the lags and forecast horizon
        return len(self.data) - max(self.lags) - self.forecast_horizon

    def __getitem__(self, idx):
        # Extract lagged inputs
        lagged_features = [
            self.data.iloc[idx - lag][
                ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos", "Year_Scaled", "DA"]
            ].values
            for lag in self.lags
        ]
        x_past = torch.tensor(lagged_features, dtype=torch.float32)

        # Extract target for forecast horizon
        y = self.data.iloc[idx : idx + self.forecast_horizon]["DA"].values
        y = torch.tensor(y, dtype=torch.float32)

        return x_past, y

# def quantile_loss(y_pred, y_true, quantiles):
#     """
#     Compute the quantile loss.

#     Parameters:
#     - y_pred (torch.Tensor): Predicted values, shape (batch_size, quantiles, time_steps)
#     - y_true (torch.Tensor): True values, shape (batch_size, time_steps)
#     - quantiles (list of float): Quantiles to consider, e.g., [0.1, 0.5, 0.9]

#     Returns:
#     - loss (torch.Tensor): Scalar tensor representing the quantile loss.
#     """
#     losses = []
#     for i, q in enumerate(quantiles):
#         errors = y_true.unsqueeze(1) - y_pred[:, i, :]
#         # Apply quantile loss formula
#         losses.append(torch.max((q - 1) * errors, q * errors).mean())

#     return torch.stack(losses).mean()  # Average over all quantiles

    

############################################
#       Tamas's expermiental models
############################################

class LSTM_Tamas(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, past_horizons, forecast_horizon, attention_heads, quantiles):
        super().__init__()
        self.past_horizons = past_horizons
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.quantiles = quantiles

        # Encoder: LSTM for historical data
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        # Decoder: Bi-directional LSTM for future data
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)

        # Temporal Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)

        # Multimodal Fusion
        self.fusion_layer = nn.Linear(hidden_size * past_horizons, hidden_size)

        # Prediction MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, forecast_horizon * quantiles)
        )

    def forward(self, x_past, x_future):
        encoded_past, _ = self.encoder(x_past)
        decoded_future, _ = self.decoder(x_future)
        decoded_future = decoded_future[:, :, :self.hidden_size] + decoded_future[:, :, self.hidden_size:]
        attention_output, _ = self.attention(decoded_future, encoded_past, encoded_past)
        fused_context = attention_output.reshape(attention_output.shape[0], -1)
        fused_context = self.fusion_layer(fused_context)
        forecast = self.mlp(fused_context).reshape(-1, self.quantiles, self.forecast_horizon)
        return forecast

def quantile_loss(y_pred, y_true, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true.unsqueeze(1) - y_pred[:, i, :]
        losses.append(torch.max((q - 1) * errors, q * errors).mean())
    return torch.stack(losses).mean()

class LaggedPriceDataset_Tamas(Dataset):
    def __init__(self, data, lags, forecast_horizon):
        self.data = data
        self.lags = lags
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - max(self.lags) - self.forecast_horizon

    def __getitem__(self, idx):
        lagged_features = [
            self.data.iloc[idx - lag][
                ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos", "Year_Scaled", "DA"]
            ].values
            for lag in self.lags
        ]
        x_past = torch.tensor(lagged_features, dtype=torch.float32)
        y = self.data.iloc[idx : idx + self.forecast_horizon]["DA"].values
        return x_past, torch.tensor(y, dtype=torch.float32)

def train_lagged_forecaster(model_class, data, lags, input_size, hidden_size, num_layers, dropout, forecast_horizon, attention_heads, quantiles, num_epochs, batch_size, optimizer_class, learning_rate, device="cpu"):
    dataset = LaggedPriceDataset_Tamas(data, lags=lags, forecast_horizon=forecast_horizon)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model_class(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        past_horizons=len(lags),
        forecast_horizon=forecast_horizon,
        attention_heads=attention_heads,
        quantiles=len(quantiles),
    ).to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_past, y in train_loader:
            x_past, y = x_past.to(device), y.to(device)
            optimizer.zero_grad()
            forecasts = model(x_past, None)
            loss = quantile_loss(forecasts, y, quantiles)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    return model

def predict(model, test_loader, device="cpu"):
    model.eval()
    all_forecasts, all_true_values = [], []
    with torch.no_grad():
        for x_past, y in test_loader:
            x_past = x_past.to(device)
            forecasts = model(x_past, None)
            all_forecasts.append(forecasts.cpu())
            all_true_values.append(y)
    return torch.cat(all_forecasts, dim=0), torch.cat(all_true_values, dim=0)

def plot_quantile_forecasts(forecasts, true_values, quantiles=[0.3, 0.5, 0.7], forecast_horizon=24):
    q_30, q_50, q_70 = forecasts[:, 0, :].detach().numpy(), forecasts[:, 1, :].detach().numpy(), forecasts[:, 2, :].detach().numpy()
    true_values = true_values.detach().numpy()
    time_steps = np.arange(forecast_horizon)

    plt.figure(figsize=(10, 6))
    plt.fill_between(time_steps, q_30[0], q_70[0], color='gray', alpha=0.3, label="0.3 - 0.7 Quantile Range")
    plt.plot(time_steps, q_50[0], label="0.5 Quantile (Median)", color='blue', linewidth=2)
    plt.plot(time_steps, true_values[0], label="True Values", color='black', linestyle='--', linewidth=2)
    plt.title("Quantile Forecasts vs True Values")
    plt.xlabel("Time Step")
    plt.ylabel("DA Price")
    plt.legend()
    plt.grid(True)
    plt.show()

