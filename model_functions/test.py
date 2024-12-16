import torch
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleLSTM
from lstm_functions import train_and_val, predict_model, plot_forecasts
from data_prep import prepare_data

import os
os.chdir(r'C:\Users\tamas\Documents\GitHub\deeplearning\formatted_data')

file_path = 'formatted_data.csv'
dense_lookback = 48  # Last 48 hours
spaced_lookback = 336  # Past 2 weeks (every 24 hours for spaced lookback)
forecast_horizon = 24  # Predict next 24 hours
gas_col = "Price_EUR_MWh"
da_col = "DA"
id_col = "ID"
future_cols = ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]
quantiles = [0.3, 0.5, 0.7]

# Prepare datasets
train_past, train_future, train_targets, val_past, val_future, val_targets, test_past, test_future, test_targets, scalers = prepare_data(
    file_path=file_path,
    dense_lookback=dense_lookback,
    spaced_lookback=spaced_lookback,
    forecast_horizon=forecast_horizon,
    future_cols=future_cols,
    gas_col=gas_col,
    da_col=da_col,
    id_col=id_col,
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(train_past), torch.tensor(train_future), torch.tensor(train_targets)),
    batch_size=64,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(val_past), torch.tensor(val_future), torch.tensor(val_targets)),
    batch_size=64,
    shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(test_past), torch.tensor(test_future), torch.tensor(test_targets)),
    batch_size=1,
    shuffle=False,
)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM(
    past_input_size=train_past.shape[2],
    future_input_size=len(future_cols),
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    past_horizons=train_past.shape[1],
    forecast_horizon=forecast_horizon,
    quantiles=quantiles,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Train model
model, train_losses, val_losses = train_and_val(
    train_loader, val_loader, num_epochs=1, model=model, optimizer=optimizer, quantiles=quantiles, device=device
)
# # Predict
forecast_inv, true_inv = predict_model(model, test_loader, scalers, quantiles=[0.3, 0.5, 0.7], forecast_horizon=24, device=device)

# # Plot results
plot_forecasts(forecast_inv, true_inv, sample_idx=0, quantiles=[0.3, 0.5, 0.7], forecast_horizon=24)
