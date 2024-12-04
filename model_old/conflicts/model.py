import torch
import torch.nn as nn

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