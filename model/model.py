import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
import numpy as np
import pandas as pd
import math as m
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/My Drive/Colab Notebooks/deeplearning/models_santi"
from keras import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

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

def accuracy(forecasts,targets):
  forecasts = forecasts.numpy()
  targets = targets.numpy()
  temp = forecasts*targets/np.abs(forecasts*targets)

  temp[temp==-1]=0 # masking the -1 with 0 because the correct ones are only the value 1
  accuracy_of_sign  = np.sum(temp)/len(forecasts)

  matches = temp==1
  mismatches = temp==0
  diff_values_matches = np.abs(targets[matches] - forecasts[matches])
  diff_values_mismatches = np.abs(targets[mismatches] - forecasts[mismatches])
  avg_price_paid_mismatches = np.mean(np.abs(forecasts[mismatches])) # forecasts is the diff between the DA and ID which is what you pay if you don't forecast correctly the sign difference (mismatch)
  avg_price_recieved_matches = np.mean(np.abs(forecasts[matches])) # forecasts is the diff between the DA and ID which is what you get paid if you do forecast correctly the sign difference (match)
  std_price_paid_mismatches = np.std(np.abs(forecasts[mismatches]))
  std_price_recieved_matches = np.std(np.abs(forecasts[matches]))

  return accuracy_of_sign, np.mean(diff_values_matches), np.mean(diff_values_mismatches), avg_price_paid_mismatches,avg_price_recieved_matches, std_price_paid_mismatches, std_price_recieved_matches
