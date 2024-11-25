import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from keras import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

def create_dataset(data_series, look_back, transforms):
    
    # log transforming that data, if necessary
    if transforms[0] == True:
        dates = data_series.index
        data_series = pd.Series(np.log(data_series), index=dates)
    
    # differencing data, if necessary
    if transforms[1] == True:
        dates = data_series.index
        data_series = pd.Series(data_series - data_series.shift(1), index=dates).dropna()

    # scaling values between 0 and 1
    dates = data_series.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    data_series = pd.Series(scaled_data[:, 0], index=dates)
    
    # creating targets and features by shifting values by 'i' number of time periods
    df = pd.DataFrame()
    for i in range(look_back+1):
        label = ''.join(['t-', str(i)])
        df[label] = data_series.shift(i)
    df = df.dropna()
    
    # splitting data into train and test sets
    train = df[:-36]
    test = df[-24:]
    
    # creating target and features for training set
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    train_dates = train.index
    
    # creating target and features for test set
    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    test_dates = test.index
    
    # reshaping data into 3 dimensions for modeling with the LSTM neural net
    X_train = np.reshape(X_train, (X_train.shape[0], 1, look_back))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, look_back))
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates, scaler

def inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler, transforms):
    
    # inverse 0 to 1 scaling
    train_predict = pd.Series(scaler.inverse_transform(train_predict.reshape(-1,1))[:,0], index=train_dates)
    y_train = pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1))[:,0], index=train_dates)

    test_predict = pd.Series(scaler.inverse_transform(test_predict.reshape(-1, 1))[:,0], index=test_dates)
    y_test = pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1))[:,0], index=test_dates)
    
    # reversing differencing if log transformed as well
    if (transforms[1] == True) & (transforms[0] == True):
        train_predict = pd.Series(train_predict + np.log(data_series.shift(1)), index=train_dates).dropna()
        y_train = pd.Series(y_train + np.log(data_series.shift(1)), index=train_dates).dropna()

        test_predict = pd.Series(test_predict + np.log(data_series.shift(1)), index=test_dates).dropna()
        y_test = pd.Series(y_test + np.log(data_series.shift(1)), index=test_dates).dropna()
    
    # reversing differencing if no log transform
    elif transforms[1] == True:
        train_predict = pd.Series(train_predict + data_series.shift(1), index=train_dates).dropna()
        y_train = pd.Series(y_train + data_series.shift(1), index=train_dates).dropna()

        test_predict = pd.Series(test_predict + data_series.shift(1), index=test_dates).dropna()
        y_test = pd.Series(y_test + data_series.shift(1), index=test_dates).dropna()
      
    # reversing log transformation
    if transforms[0] == True:
        train_predict = pd.Series(np.exp(train_predict), index=train_dates)
        y_train = pd.Series(np.exp(y_train), index=train_dates)

        test_predict = pd.Series(np.exp(test_predict), index=test_dates)
        y_test = pd.Series(np.exp(y_test), index=test_dates)
        
    return train_predict, y_train, test_predict, y_test

def lstm_model(data_series, look_back, transforms, lstm_params):

    """
    Train an LSTM model and predict mean and standard deviations.
    
    Args:
    - data_series (pd.Series): Original time series data.
    - look_back (int): Number of past observations to use for predictions.
    - transforms (list): Transformations to apply to the data.
    - lstm_params (tuple): Parameters for LSTM (units, epochs, verbose).

    Returns:
    - test_mean_series (pd.Series): Predicted mean values (test set).
    - test_std_series (pd.Series): Predicted standard deviations (test set).
    - y_test_series (pd.Series): Original test set values (unfiltered).
    """

    np.random.seed(1)
    train_predict = []
    test_predict = []

    # creating the training and testing datasets
    X_train, y_train, X_test, y_test, train_dates, test_dates, scaler = create_dataset(data_series, look_back, transforms)

    # unpacking lstm_params
    units, epochs, verbose = lstm_params

    # training the model
    model = Sequential()
    model.add(LSTM(units, input_shape=(1, look_back)))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=verbose)
    
    # making predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # creating separate numpy arrays for mean and standard deviation
    train_mean = train_predict[:, 0]
    test_mean = test_predict[:, 0]
    train_std = train_predict[:, 1]
    test_std = test_predict[:, 1]

    # Ensure train_mean and train_std are reshaped for inverse_transform
    train_mean = train_mean.reshape(-1, 1)
    test_mean = test_mean.reshape(-1, 1)
    train_std = train_std.reshape(-1, 1)
    test_std = test_std.reshape(-1, 1)

    # Inverse transform means
    train_mean_series, y_train_series, test_mean_series, y_test_series = inverse_transforms(
        train_mean, y_train, test_mean, y_test, data_series, train_dates, test_dates, scaler, transforms
    )

    # Inverse transform standard deviations
    train_std_series, _, test_std_series, _ = inverse_transforms(
        train_std, y_train, test_std, y_test, data_series, train_dates, test_dates, scaler, transforms
    )

    # Plot predictions with confidence intervals
    fig, ax = plt.subplots(figsize=(12, 6))
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)

    # Plot actual values
    plt.plot(y_test_series.index, y_test_series, label="Actual Values", color="blue", alpha=0.7)

    # Plot mean predictions
    plt.plot(test_mean_series.index, test_mean_series, label="Predicted Mean", color="red", alpha=0.8)

    # Confidence intervals
    lower_bound = test_mean_series - 1.96 * test_std_series
    upper_bound = test_mean_series + 1.96 * test_std_series
    plt.fill_between(test_mean_series.index, lower_bound, upper_bound, color='orange', alpha=0.2, label="95% CI")

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Predictions with Confidence Intervals")
    plt.legend()
    plt.show()

    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train_series, train_mean_series))
    test_rmse = np.sqrt(mean_squared_error(y_test_series, test_mean_series))
    print('Train RMSE: %.3f' % train_rmse)
    print('Test RMSE: %.3f' % test_rmse)
    
    return test_mean_series, test_std_series, y_test_series

def gauss_compare(original_series, predictions_mean, predictions_std):
    """
    Compare the original series with Gaussian-filtered predictions.
    
    Args:
    - original_series (pd.Series): The original unfiltered time series data.
    - predictions_mean (pd.Series): The predicted mean values from the model.
    - predictions_std (pd.Series): The predicted standard deviations from the model.

    Returns:
    - None: Displays a plot and prints RMSE.
    """
    # Align predictions with the last part of the original series
    aligned_original = original_series[-24:]

    # Calculate confidence intervals
    lower_bound = predictions_mean - 1.96 * predictions_std
    upper_bound = predictions_mean + 1.96 * predictions_std

    # Plot the original series and predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)

    plt.plot(aligned_original.index, aligned_original, label="Original Series", color="blue", alpha=0.7)
    plt.plot(predictions_mean.index, predictions_mean, label="Predicted Mean", color="red", alpha=0.8)
    plt.fill_between(
        predictions_mean.index, lower_bound, upper_bound, color='orange', alpha=0.2, label="95% Confidence Interval"
    )
    plt.title("Gauss-Filtered Predictions vs. Original Series")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

    # Calculate RMSE
    error = np.sqrt(mean_squared_error(aligned_original, predictions_mean))
    print('Test RMSE: %.3f' % error)
