import torch
from torch.utils.data import DataLoader
from model import LSTM_Tamas, TimeSeriesDataset
from lstm_functions import prepare_data, train_and_evaluate_model, predict_model, plot_forecasts

import os
os.chdir(r'C:\Users\tamas\Documents\GitHub\deeplearning\formatted_data')

file_path = 'formatted_data.csv'
lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
        38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 72, 96, 120, 144, 168, 336, 504, 672, 840, 1008, 1344, 1680]
gas_col = "Price_EUR_MWh"
da_col = "DA"
id_col = "ID"
temporal_cols = ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]
forecast_horizon = 24
quantiles = [0.5]

# Prepare data
train_data, test_data, scalers = prepare_data(file_path, lags, temporal_cols, gas_col, da_col, id_col, forecast_horizon=forecast_horizon)

# Initialize datasets and loaders
train_dataset = TimeSeriesDataset(train_data, lags, forecast_horizon, temporal_cols, gas_col, da_col, id_col)
test_dataset = TimeSeriesDataset(test_data, lags, forecast_horizon, temporal_cols, gas_col, da_col, id_col)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Tamas(past_input_size=train_data.shape[1],
                                future_input_size=len(temporal_cols),
                                hidden_size=128,
                                num_layers=5,
                                dropout=0.2,
                                past_horizons=48,
                                forecast_horizon=forecast_horizon,
                                quantiles=quantiles,
                                attention_heads=4,
                                num_periods=3).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Train model
model, tr_loss, val_loss = train_and_evaluate_model(model, train_loader, test_loader, optimizer, quantiles, device=device, num_epochs=10)

# # Predict
forecasts, true_values = predict_model(model, test_loader, scalers, da_col, quantiles, forecast_horizon, device)

# # Plot results
plot_forecasts(forecasts, true_values, sample_idx=0, quantiles=quantiles, forecast_horizon=forecast_horizon)