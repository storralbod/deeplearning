import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class LSTM_multivariate_input_multi_step_forecaster(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,dropout, past_horizon, forecast_horizon, future_inputs_size):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.future_inputs_size = future_inputs_size
        self.mlp_input_size = hidden_size+future_inputs_size*forecast_horizon
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,dropout=dropout,batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size,48*4),
            nn.ReLU(),
            nn.LayerNorm(48*4),
            nn.Dropout(dropout),
            nn.Linear(48*4,48*3),
            nn.ReLU(),
            nn.LayerNorm(48*3),
            nn.Dropout(dropout),
            nn.Linear(48*3,48*3),
            nn.ReLU(),
            nn.LayerNorm(48*3),
            nn.Dropout(dropout),
            nn.Linear(48*3,forecast_horizon)
        )

      #### new expanded mlp with future features ######
        self.mlp_expanded = nn.Sequential(
            nn.Linear(self.mlp_input_size,self.mlp_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size*2,self.mlp_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size*2,self.mlp_input_size*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size*2,24)
        )
        def mlp_per_hour(h):
          self.mlp_input_size_temp = hidden_size+future_inputs_size*(h+1)
          self.mlp_expanded_temp = nn.Sequential(
            nn.Linear(self.mlp_input_size_temp,self.mlp_input_size_temp*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size_temp*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size_temp*2,self.mlp_input_size_temp*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size_temp*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size_temp*2,self.mlp_input_size_temp*2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_input_size_temp*2),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_input_size_temp*2,1)
        )
          return self.mlp_expanded_temp

        self.all_hourly_mlps = [self.mlp_expanded for h in range(forecast_horizon)]

    def forward(self,x,future_inputs):

        if torch.isnan(x).any():
          nans = torch.isnan(x).any().sum()
          raise ValueError(f"Input contains {nans} NaN values")
        
        lstm_out, (hidden, c) = self.lstm(x)
        #forecast = self.mlp(hidden[-1,:,:])

        #### Exapnding to include future features ####

        # 1. Easy decoder 
        context = hidden[-1,:,:]
        #future_input = torch.cat([context,future_inputs.view(future_inputs.shape[0],-1)],axis=1)
        #forecasts = self.mlp_expanded(future_input)

        #return forecasts.unsqueeze(-1)

        # 2. More complete decoder
        forecasts = []
        for h in range(self.forecast_horizon):
          future_input = torch.cat([context,future_inputs[:,:h,:]],axis=1) #[B,feature_size_length=104]
          forecast = self.all_hourly_mlps[h](future_input)
          forecasts.append(forecast)

        forecasts = torch.cat(forecasts,axis=1).unsqueeze(-1)
        return forecasts
    
###########################################
#       Tamas's experimental model
###########################################
class SimpleLSTM(nn.Module):
    def __init__(self, 
                 past_input_size, 
                 future_input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout, 
                 past_horizons, 
                 forecast_horizon, 
                 quantiles):
        super().__init__()
        self.past_horizons = past_horizons
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.quantiles = quantiles

        # LSTM Encoder for past data
        self.encoder = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # LSTM Decoder for future data
        self.decoder = nn.LSTM(
            input_size=future_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Output layer for quantiles
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_size, forecast_horizon) for _ in range(len(quantiles))
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

    def forward(self, x_past, x_future):
        """
        Forward pass for the model.
        Args:
        - x_past: Tensor of shape [batch_size, past_horizons, past_input_size]
        - x_future: Tensor of shape [batch_size, forecast_horizon, future_input_size]
        Returns:
        - quantile_forecasts: Tensor of shape [batch_size, len(quantiles), forecast_horizon]
        """
        # Encode past data
        _, (hidden_state, _) = self.encoder(x_past)  # Use the hidden state from the encoder

        # Decode future data, initialized with the encoder's hidden state
        decoder_output, _ = self.decoder(x_future, (hidden_state, torch.zeros_like(hidden_state)))

        # Predict quantiles using separate output heads
        quantile_forecasts = []
        for quantile_head in self.quantile_heads:
            quantile_forecasts.append(quantile_head(decoder_output[:, -1, :]))  # Use the last decoder output

        # Stack quantile forecasts
        quantile_forecasts = torch.stack(quantile_forecasts, dim=1)  # [batch_size, len(quantiles), forecast_horizon]

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