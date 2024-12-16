import pandas as pd 
import numpy as np
import glob
import torch
from sklearn.preprocessing import MinMaxScaler

# Function to encode time (day, month, hour) as sine and cosine
def encode_time(value, max_value):
    value_sin = np.sin(2 * np.pi * value / max_value)
    value_cos = np.cos(2 * np.pi * value / max_value)
    return value_sin, value_cos


def create_dataframe():
    # File paths for marginalpdbc and precious_pibcic files
    marginalpdbc_files = glob.glob('../raw_data/marginalpdbc*.csv')
    precious_files = glob.glob('../raw_data/precios_pibci*.csv')

    print(f"Found {len(marginalpdbc_files)} marginalpdbc files")
    print(f"Found {len(precious_files)} precious files")

    # Process marginalpdbc files
    marginal_data = []
    for file in marginalpdbc_files:
        #print(f"Processing marginalpdbc file: {file}")
        try:
            data = pd.read_csv(file, delimiter=';', header=None, skiprows=1, usecols=range(6), encoding='latin1').iloc[:-1, :]
            data.columns = ['Year', 'Month', 'Day', 'Hour', 'Price1', 'Unused']
            data = data[['Year', 'Month', 'Day', 'Hour', 'Price1']].dropna()

            # Convert columns to numeric
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
            data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Price1'] = pd.to_numeric(data['Price1'], errors='coerce')
            data = data.dropna(subset=['Year', 'Month', 'Day', 'Hour', 'Price1'])

            # Encode time features
            data['Hour_Sin'], data['Hour_Cos'] = zip(*data['Hour'].apply(lambda x: encode_time(x, 24)))
            data['Day_Sin'], data['Day_Cos'] = zip(*data['Day'].apply(lambda x: encode_time(x, 31)))
            data['Month_Sin'], data['Month_Cos'] = zip(*data['Month'].apply(lambda x: encode_time(x, 12)))

            # Scale Year
            data['Year_Scaled'] = (data['Year'] - 2018) * 0.1 + 0.1

            marginal_data.append(data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Combine all marginalpdbc data
    if marginal_data:
        marginal_data = pd.concat(marginal_data, ignore_index=True)
        print(f"Processed {len(marginal_data)} rows from marginalpdbc files")
    else:
        print("No valid marginalpdbc data processed.")

    # Process precious_pibcic files
    precious_data = []
    for file in precious_files:
        #print(f"Processing precious file: {file}")
        try:
            data = pd.read_csv(file, delimiter=';', skiprows=2, encoding='latin1')  # Skip first two metadata rows
            data = data.rename(columns=lambda x: x.strip())  # Normalize column names
            data = data.rename(columns={
                'Año': 'Year',
                'Mes': 'Month',
                'Día': 'Day',
                'Hora': 'Hour',
                'MedioES': 'Price2'
            })[['Year', 'Month', 'Day', 'Hour', 'Price2']].dropna()

            # Convert columns to numeric
            data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
            data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
            data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
            data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
            data['Price2'] = pd.to_numeric(data['Price2'].str.replace(',', '.'), errors='coerce')  # Handle decimal commas
            data = data.dropna(subset=['Year', 'Month', 'Day', 'Hour', 'Price2'])

            precious_data.append(data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Combine all precious_pibcic data
    if precious_data:
        precious_data = pd.concat(precious_data, ignore_index=True)
        print(f"Processed {len(precious_data)} rows from precious files")
    else:
        print("No valid precious data processed.")

    # Merge marginalpdbc and precious_pibcic data on Year, Month, Day, and Hour
    combined_data = pd.merge(
        marginal_data,
        precious_data,
        on=['Year', 'Month', 'Day', 'Hour'],
        how='inner',
        suffixes=('_marginal', '_precious')
    )
    return combined_data


def discrete_prices(prices,delta):

    return np.arange(int(min(prices)),int(max(prices))+1,delta)

# One-hot encoding price labels
def one_hot_encoding(prices,discrete_prices) -> np.array:

    one_hot_prices = []
    for price in prices:
        one_hot = np.zeros(len(discrete_prices))
        idx = np.argmin(abs(price-discrete_prices))
        one_hot[idx] = 1.0
        one_hot_prices.append(one_hot)

    return np.array(one_hot_prices)

# log transform of input and output data
def log_transform(values):
    shifted_values = values + 100  # adding a small epsilon to avoid log(0)
    shifted_values = np.array(shifted_values)
    transformed = np.log(np.abs(shifted_values) + 1e-10)  # adding a small epsilon to avoid log(0)
    transformed[values == 0] = 0
    return transformed

def minmaxscaler(inputs):

  scaler = MinMaxScaler(feature_range=(0,1))
  scaler.fit(inputs)
  inputs_scaled = scaler.transform(inputs)

  return scaler,inputs_scaled.ravel()

def create_features(input_features, target, dense_lookback, spaced_lookback, forecast_horizon, future_indices):
    """
    Create features with both dense and spaced lookback for short- and long-term patterns.
    """
    past_inputs, future_inputs, outputs = [], [], []

    # Create indices for spaced lookback
    spaced_steps = np.arange(0, spaced_lookback, step=24)  # Example: every 24 timesteps (1 day)

    for i in range(len(input_features) - max(dense_lookback, spaced_lookback) - forecast_horizon):
        # Extract dense past features
        dense_past_features = input_features[i : i + dense_lookback]

        # Extract spaced past features
        spaced_past_features = input_features[i + spaced_steps]

        # Combine dense and spaced features
        past_features = np.concatenate([dense_past_features, spaced_past_features], axis=0)

        # Extract target values for the forecast horizon
        target_slice = target[i + dense_lookback : i + dense_lookback + forecast_horizon]

        # Ensure target slice has the correct length
        if len(target_slice) != forecast_horizon:
            continue

        # Extract future features, if specified
        if future_indices is not None:
            future_features = input_features[i + dense_lookback : i + dense_lookback + forecast_horizon, future_indices]
            # Skip iteration if future_features is incomplete
            if future_features.shape[0] != forecast_horizon:
                continue
        else:
            future_features = np.zeros((forecast_horizon, input_features.shape[1]))  # Placeholder

        # Append to the lists
        past_inputs.append(past_features)
        future_inputs.append(future_features)
        outputs.append(target_slice)

    # Convert lists to NumPy arrays
    past_inputs = np.array(past_inputs)  # Shape: [samples, lookback_steps, features]
    future_inputs = np.array(future_inputs)  # Shape: [samples, forecast_horizon, future_features]
    outputs = np.array(outputs)  # Shape: [samples, forecast_horizon, targets]

    return past_inputs, future_inputs, outputs


def create_datasets(features, target, dense_lookback, spaced_lookback, forecast_horizon, future_cols=None, p_train=0.7, p_val=0.2, p_test=0.1):
    """
    Create datasets with dense and spaced lookback for LSTM training.
    """
    assert len(features) == len(target), "Features and target must have the same length"

    hours = len(features)

    # Extract the indices of the future columns
    if future_cols is not None:
        future_indices = future_cols
    else:
        future_indices = None

    # Set dataset sizes
    usable_hours = hours - max(dense_lookback, spaced_lookback) - forecast_horizon
    num_train = int(usable_hours * p_train)
    num_val = int(usable_hours * p_val)
    num_test = usable_hours - num_train - num_val

    # Generate features and labels
    past_inputs, future_inputs, outputs = create_features(
        features, target, dense_lookback, spaced_lookback, forecast_horizon, future_indices
    )

    # Split into train, val, and test sets
    train_cutoff = num_train
    val_cutoff = num_train + num_val

    train_past = past_inputs[:train_cutoff]
    val_past = past_inputs[train_cutoff:val_cutoff]
    test_past = past_inputs[val_cutoff:]

    train_future = future_inputs[:train_cutoff]
    val_future = future_inputs[train_cutoff:val_cutoff]
    test_future = future_inputs[val_cutoff:]

    train_targets = outputs[:train_cutoff]
    val_targets = outputs[train_cutoff:val_cutoff]
    test_targets = outputs[val_cutoff:]

    return train_past, train_future, train_targets, val_past, val_future, val_targets, test_past, test_future, test_targets


def prepare_data(file_path, dense_lookback, spaced_lookback, forecast_horizon, future_cols, gas_col, da_col, id_col):
    """
    Prepare data for forked training with dense and spaced lookback.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        dense_lookback (int): Number of consecutive timesteps for dense lookback.
        spaced_lookback (int): Maximum lookback period for spaced lookback.
        forecast_horizon (int): Number of timesteps to forecast.
        future_cols (list): Names of future columns.
        gas_col, da_col, id_col (str): Column names for specific features.
    Returns:
        Tuple of train, validation, test datasets and scalers (feature scalers and target scaler).
    """
    # Load and preprocess data
    data = pd.read_csv(file_path)
    data = data.dropna().reset_index(drop=True)
    data = data.drop(columns=["Volume_MWh", "Diff", "Year", "Month", "Day", "Hour"])

    # Prepare target and feature indices
    output_data = np.array(data.loc[:, da_col]).reshape(-1, 1)
    future_indices = [data.columns.get_loc(col) for col in future_cols]

    # Scaling preparation
    scale_cols = [da_col, id_col, gas_col]
    scale_indices = [data.columns.get_loc(col) for col in scale_cols]

    # Convert to NumPy array for faster processing
    data = np.array(data)

    # Create past and future datasets with dense and spaced lookback
    train_past, train_future, train_targets, val_past, val_future, val_targets, test_past, test_future, test_targets = create_datasets(
        features=data,
        target=output_data,
        dense_lookback=dense_lookback,
        spaced_lookback=spaced_lookback,
        forecast_horizon=forecast_horizon,
        future_cols=future_indices,
        p_train=0.7,
        p_val=0.2,
        p_test=0.1,
    )

    # Initialize scalers for features
    scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in scale_cols}

    # Apply scaling to past features only
    for col, idx in zip(scale_cols, scale_indices):
        # Fit the scaler on the training past data (flattened)
        flat_train_past = train_past[:, :, idx].flatten().reshape(-1, 1)
        scalers[col].fit(flat_train_past)

        # Transform train, val, and test datasets using the same scaler
        for dataset in [train_past, val_past, test_past]:
            dataset[:, :, idx] = scalers[col].transform(
                dataset[:, :, idx].reshape(-1, 1)
            ).reshape(dataset.shape[0], dataset.shape[1])

    # Scale the targets (output_data)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).reshape(train_targets.shape)
    val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).reshape(val_targets.shape)
    test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).reshape(test_targets.shape)

    # Convert datasets to PyTorch tensors 32
    train_past = torch.tensor(train_past, dtype=torch.float32)
    train_future = torch.tensor(train_future, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)

    val_past = torch.tensor(val_past, dtype=torch.float32)
    val_future = torch.tensor(val_future, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)

    test_past = torch.tensor(test_past, dtype=torch.float32)
    test_future = torch.tensor(test_future, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    # Return datasets and scalers
    return (
        train_past,
        train_future,
        train_targets,
        val_past,
        val_future,
        val_targets,
        test_past,
        test_future,
        test_targets,
        target_scaler,
    )