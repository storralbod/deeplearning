import pandas as pd
import lstm_functions as lf
import datetime as dt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# oil_data = yf.download('CL=F', start='2023-01-01', end='2023-12-31')
# oil_data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

data = pd.read_csv('model/merged_marginal_data.csv')

data['Date'] = data.apply(
    lambda row: (
        dt.datetime(row['Year'].astype(int), row['Month'].astype(int), row['Day'].astype(int), 23, 59, 59)
        if row['Hour'].astype(int) == 24 or row['Hour'].astype(int) == 25
        else dt.datetime(row['Year'].astype(int), row['Month'].astype(int), row['Day'].astype(int), row['Hour'].astype(int))
    ), 
    axis=1
)

data.drop(['Year', 'Month', 'Day', 'Hour'], axis=1, inplace=True)
data.set_index('Date', inplace=True)
data = data.sort_index()

df = data['Price1']

look_back = 1
split = 0.7
log = True
difference = True
transforms = [log, difference]

nodes = 4
epochs = 2
verbose = 0 # 0=print no output, 1=most, 2=less, 3=least
lstm_params = [nodes, epochs, verbose]

train_predict, y_train, test_predict, y_test = lf.lstm_model(df, look_back, split, transforms, lstm_params)

gaussian_filtering = pd.Series(gaussian_filter(df, sigma=1), index=df.index).astype(float)
data.set_index('Date', inplace=True)
data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
data = data.sort_index()

df = data['Price1']

look_back = 1
split = 0.7
log = True
difference = True
transforms = [log, difference]

nodes = 4
epochs = 2
verbose = 0 # 0=print no output, 1=most, 2=less, 3=least
lstm_params = [nodes, epochs, verbose]

train_predict, y_train, test_predict, y_test = lf.lstm_model(df, look_back, split, transforms, lstm_params)

gaussian_filtering = pd.Series(gaussian_filter(df, sigma=1), index=df.index).astype(float)

# running LSTM with Gaussian-filtered data
look_back = 1
split = 0.7
log = True
difference = True
transforms = [log, difference]

nodes = 4
epochs = 50
verbose = 0 # 0=print no output, 1=most, 2=less, 3=least
lstm_params = [nodes, epochs, verbose]

train_predict, y_train, test_predict, y_test = lf.lstm_model(gaussian_filtering, look_back, split, transforms, lstm_params)

# comparing gaussian model results to original data
lf.gauss_compare(df, test_predict, split)