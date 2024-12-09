import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

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
    
###########################################
#       Tamas's experimental model
###########################################
import torch
import torch.nn as nn


class LSTM_Tamas(nn.Module):
    def __init__(self, 
                 past_input_size, 
                 future_input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout, 
                 past_horizons, 
                 forecast_horizon, 
                 quantiles, 
                 attention_heads, 
                 num_periods=3):
        super().__init__()
        self.past_horizons = past_horizons
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.quantiles = quantiles
        self.num_periods = num_periods

        # Projection layers for past and future inputs
        self.past_projection = nn.Linear(past_input_size, hidden_size)
        self.future_projection = nn.Linear(future_input_size, hidden_size)

        # Encoder LSTM for past data
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Decoder Bi-LSTM for future data
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # Attention layers for different time horizons
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attention_heads, batch_first=True)
            for _ in range(num_periods)
        ])
        self.attention_dropout = nn.Dropout(dropout)

        # Fusion layer for concatenated attention outputs
        self.fusion_layer = nn.Linear(hidden_size * num_periods + hidden_size, hidden_size)

        # Separate MLP heads for each quantile to reduce interference
        self.mlp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 3),
                nn.ReLU(),
                nn.LayerNorm(hidden_size * 3),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 3, forecast_horizon)
            )
            for _ in range(len(quantiles))
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.zeros_(param.data)

    def forward(self, x_past, x_future=None):
        # Project past inputs
        x_past_projected = self.past_projection(x_past)  # [batch, past_horizons, hidden_size]

        # Project future inputs or initialize as zeros if not provided
        if x_future is not None:
            x_future_projected = self.future_projection(x_future)  # [batch, forecast_horizon, hidden_size]
        else:
            batch_size = x_past.size(0)
            x_future_projected = torch.zeros(batch_size, self.forecast_horizon, self.hidden_size, device=x_past.device)

        # Encode past inputs
        encoded_past, _ = self.encoder(x_past_projected)  # [batch, past_horizons, hidden_size]

        # Decode future inputs
        decoded_future, _ = self.decoder(x_future_projected)  # [batch, forecast_horizon, hidden_size*2]
        decoded_future = decoded_future[:, :, :self.hidden_size] + decoded_future[:, :, self.hidden_size:]  # Combine bidirectional outputs

        # Dynamically adjust window sizes based on sequence length
        seq_len = encoded_past.size(1)
        hourly_window = min(48, seq_len)
        daily_window = min(168, seq_len)
        monthly_window = min(720, seq_len)

        # Extract subsets of past encodings for attention
        hourly_slice = encoded_past[:, -hourly_window:, :] if hourly_window > 0 else encoded_past
        daily_slice = encoded_past[:, -daily_window:, :] if daily_window > 0 else encoded_past
        monthly_slice = encoded_past[:, -monthly_window:, :] if monthly_window > 0 else encoded_past

        slices = [hourly_slice, daily_slice, monthly_slice]

        # Apply attention to each time period slice
        attention_outputs = []
        for attention_layer, slice_data in zip(self.attention_layers, slices):
            attn_output, _ = attention_layer(decoded_future, slice_data, slice_data)
            attn_output = self.attention_dropout(attn_output)
            attention_outputs.append(attn_output + decoded_future)  # Residual connection

        # Concatenate attention outputs and fuse with decoded_future
        fused_context = torch.cat(attention_outputs, dim=-1)  # [batch, forecast_horizon, hidden_size * num_periods]
        fused_context = torch.cat([fused_context, decoded_future], dim=-1)  # Add decoded future context
        fused_context = self.fusion_layer(fused_context)  # [batch, forecast_horizon, hidden_size]

        # Predict quantiles using separate MLP heads
        quantile_forecasts = []
        for mlp_head in self.mlp_heads:
            quantile_forecasts.append(mlp_head(fused_context))  # [batch, forecast_horizon]

        # Stack quantile forecasts
        quantile_forecasts = torch.stack(quantile_forecasts, dim=1)  # [batch, quantiles, forecast_horizon]
        return quantile_forecasts


# Temporal Attention Class
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        # Compute attention scores
        scores = self.attn_weights(lstm_outputs).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]

        # Compute context vector
        context = torch.sum(lstm_outputs * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size]
        return context, attn_weights

# ----------------------------------
# Define the TimeSeriesDataset Class
# ----------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 data, 
                 lags, 
                 forecast_horizon, 
                 temporal_cols=None, 
                 gas_col="Price_EUR_MWh", 
                 da_col="DA",
                 id_col="ID"):
        """
        Dataset for time-series forecasting with custom lag steps and fixed gas lags at t-24 and t-48.
        
        Parameters
        ----------
        data : pd.DataFrame
            The complete dataset with at least:
            - DA (day-ahead price)
            - ID (intraday price)
            - gas_col (gas price, e.g. Price_EUR_MWh)
            - temporal_cols (e.g., Hour_Sin, Hour_Cos, etc.)
        
        lags : list of int
            List of custom lagged timesteps (in hours) to include as sequence steps for DA and ID.
        
        forecast_horizon : int
            Number of future timesteps to predict.
        
        temporal_cols : list of str, optional
            Columns representing temporal features.
        
        gas_col : str, optional
            Column for gas feature.
        
        da_col : str, optional
            Column name of the DA target variable (and feature).
        
        id_col : str, optional
            Column name of the ID variable.
        """
        
        self.data = data.reset_index(drop=True)
        self.lags = sorted(lags)  # ensure sorted order
        self.forecast_horizon = forecast_horizon
        self.gas_col = gas_col
        self.da_col = da_col
        self.id_col = id_col

        if temporal_cols is None:
            temporal_cols = ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]
        self.temporal_cols = temporal_cols

        # Ensure we have enough history for largest lag and for gas lags (t-24, t-48)
        self.start_index = max(max(self.lags), 48)  # Ensure at least 48 steps back for gas
        self.end_index = len(self.data) - self.forecast_horizon

    def __len__(self):
        return self.end_index - self.start_index

    def __getitem__(self, idx):
        idx = idx + self.start_index
        
        # Retrieve gas features at t-24 and t-48 from current index
        gas_24 = self.data.iloc[idx - 24][self.gas_col]
        gas_48 = self.data.iloc[idx - 48][self.gas_col]

        # Construct x_past
        # For each lag, we take:
        # - temporal features from data at t-lag
        # - DA and ID at t-lag
        # - gas_24 and gas_48 are appended to each time step to keep dimensionality consistent
        past_steps = []
        for lag in self.lags:
            row = self.data.iloc[idx - lag]
            temporal_features = row[self.temporal_cols].values.astype(np.float32)

            da_value = row[self.da_col]
            id_value = row[self.id_col]

            # Combine all features for this timestep
            step_features = np.concatenate([
                temporal_features, 
                np.array([da_value, id_value], dtype=np.float32),
                np.array([gas_24, gas_48], dtype=np.float32)  # Append the two gas features
            ], axis=0)
            past_steps.append(step_features)

        x_past = torch.tensor(np.array(past_steps), dtype=torch.float32)  # shape: [len(lags), 10]

        # Construct x_future
        future_rows = self.data.iloc[idx: idx + self.forecast_horizon]
        x_future = torch.tensor(future_rows[self.temporal_cols].values.astype(np.float32), dtype=torch.float32)
        # shape: [forecast_horizon, 6]

        # Construct y (target)
        y = torch.tensor(future_rows[self.da_col].values.astype(np.float32), dtype=torch.float32)
        # shape: [forecast_horizon]

        return x_past, x_future, y