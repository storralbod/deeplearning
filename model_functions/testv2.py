import torch
from torch.utils.data import DataLoader
from model import LSTM_multivariate_input_multi_step_forecaster, LSTM_QuantileForecaster
from lstm_functionsv2 import prepare_data_with_weather, train_and_evaluate_model, predict_model, plot_forecasts

# Define column groups
historical_columns = ['Price_EUR_MWh', 'Volume_MWh', 'ID']
predictive_columns = [
    'Andalusia (Wind)_temperature', 'Andalusia (Wind)_wind_u_component', 'Andalusia (Wind)_wind_v_component',
    'Aragon (Wind)_temperature', 'Aragon (Wind)_wind_u_component', 'Aragon (Wind)_wind_v_component',
    'Madrid_temperature', 'Madrid_wind_u_component', 'Madrid_wind_v_component'
]
temporal_cols = ['Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos']
target_column = 'DA'

# Prepare data
train_loader, test_loader, scaler_historical, scaler_predictive, scaler_target, input_size = prepare_data_with_weather(
    file_path='formatted_data/formatted_data.csv',
    historical_columns=historical_columns,
    predictive_columns=predictive_columns,
    temporal_cols=temporal_cols,
    target_column=target_column,
    sequence_length=336,
    forecast_horizon=24,
    test_split_ratio=0.1
)

# Set future_inputs_size to match the predictive columns
future_inputs_size = len(predictive_columns) + len(temporal_cols)

# Define model
model = LSTM_QuantileForecaster(
    input_size=input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.1,
    past_horizon=336,
    forecast_horizon=24,
    future_inputs_size=len(predictive_columns),
    quantiles=[0.1, 0.5, 0.9]
)

# Training and evaluation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the model
model, train_losses, val_losses = train_and_evaluate_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    optimizer=optimizer,
    quantiles=[0.1, 0.5, 0.9],
    device=device,
    num_epochs=10
)

# Predict and plot
forecast_inv, true_inv = predict_model(
    model=model,
    test_loader=test_loader,
    scaler_target=scaler_target,
    quantiles=[0.1, 0.5, 0.9],
    forecast_horizon=24,
    device=device
)
plot_forecasts(forecast_inv, true_inv, sample_idx=0, quantiles=[0.1, 0.5, 0.9], forecast_horizon=24)
