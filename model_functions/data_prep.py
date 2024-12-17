import pandas as pd 
import numpy as np
import glob
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

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

def create_features(input_features, target, dense_lookback, spaced_lookback, forecast_horizon, future_indices=None, step_growth_factor=None):
    """
    Create features with both dense and optionally spaced lookback for short- and long-term patterns.
    
    Args:
    - input_features (np.ndarray): Feature data.
    - target (np.ndarray): Target variable.
    - dense_lookback (int): Number of consecutive recent timesteps for dense lookback.
    - spaced_lookback (int): Maximum lookback period (e.g., 2 years). Set to 0 to skip spaced lookback.
    - forecast_horizon (int): Number of timesteps to forecast.
    - future_indices (list[int], optional): Indices of future features.
    - step_growth_factor (float, optional): If provided, creates dynamic spaced steps with growing intervals.
    
    Returns:
    - dense_past_inputs (np.ndarray): Dense lookback features. Shape: [samples, dense_lookback, features].
    - spaced_past_inputs (np.ndarray): Spaced lookback features. Shape: [samples, spaced_steps, features] or None.
    - future_inputs (np.ndarray): Future features for the decoder. Shape: [samples, forecast_horizon, future_features].
    - outputs (np.ndarray): Forecast horizon target values. Shape: [samples, forecast_horizon, targets].
    """
    dense_past_inputs, spaced_past_inputs, future_inputs, outputs = [], [], [], []

    # Generate spaced steps
    if spaced_lookback > 0:
        if step_growth_factor is not None:
            # Dynamic step generation
            spaced_steps = [0]  # Start at timestep t
            step_size = 24  # Start with 1 day (24 timesteps)
            total_steps = 0

            while total_steps + step_size < spaced_lookback:
                total_steps += step_size
                spaced_steps.append(total_steps)
                step_size = int(step_size * step_growth_factor)  # Increase step size

            spaced_steps = np.array(spaced_steps)
        else:
            # Fixed step size
            spaced_steps = np.arange(0, spaced_lookback, step=24)
    else:
        spaced_steps = None

    for i in range(len(input_features) - dense_lookback - (max(spaced_steps) if spaced_steps is not None else 0) - forecast_horizon):
        # Extract dense past features
        dense_past_features = input_features[i : i + dense_lookback]

        # Extract spaced past features if applicable
        if spaced_steps is not None:
            spaced_past_indices = i + dense_lookback - spaced_steps[::-1]  # Reverse for chronological order
            spaced_past_features = input_features[spaced_past_indices]
        else:
            spaced_past_features = None

        # Extract target values for the forecast horizon
        target_slice = target[i + dense_lookback : i + dense_lookback + forecast_horizon]
        if len(target_slice) != forecast_horizon:
            continue  # Skip incomplete slices

        # Extract future features
        if future_indices is not None:
            future_features = input_features[i + dense_lookback : i + dense_lookback + forecast_horizon, future_indices]
        else:
            future_features = np.zeros((forecast_horizon, input_features.shape[1]))  # Placeholder if no future features

        # Append to the lists
        dense_past_inputs.append(dense_past_features)
        if spaced_past_features is not None:
            spaced_past_inputs.append(spaced_past_features)
        future_inputs.append(future_features)
        outputs.append(target_slice)

    # Convert to NumPy arrays
    dense_past_inputs = np.array(dense_past_inputs)
    spaced_past_inputs = np.array(spaced_past_inputs) if spaced_steps is not None else None
    future_inputs = np.array(future_inputs)
    outputs = np.array(outputs)

    return dense_past_inputs, spaced_past_inputs, future_inputs, outputs


def create_datasets(
    features,
    target,
    dense_lookback,
    spaced_lookback,
    forecast_horizon,
    future_cols=None,
    p_train=0.7,
    p_val=0.2,
    p_test=0.1,
    spaced=True,
    step_growth_factor=None
):
    """
    Create datasets with dense and optionally spaced lookback for LSTM training.
    """
    assert len(features) == len(target), "Features and target must have the same length"

    hours = len(features)
    future_indices = future_cols if future_cols else None

    # Set dataset sizes
    usable_hours = hours - max(dense_lookback, spaced_lookback if spaced else 0) - forecast_horizon
    num_train = int(usable_hours * p_train)
    num_val = int(usable_hours * p_val)
    num_test = usable_hours - num_train - num_val

    # Generate features and labels
    if spaced:
        dense_past, spaced_past, future_inputs, outputs = create_features(
            features,
            target,
            dense_lookback,
            spaced_lookback,
            forecast_horizon,
            future_indices,
            step_growth_factor=step_growth_factor,
        )
    else:
        spaced_past = None
        dense_past, _, future_inputs, outputs = create_features(
            features,
            target,
            dense_lookback,
            0,  # No spaced lookback
            forecast_horizon,
            future_indices,
        )

    # Split datasets
    train_dense = dense_past[:num_train]
    val_dense = dense_past[num_train : num_train + num_val]
    test_dense = dense_past[num_train + num_val :]

    if spaced:
        train_spaced = spaced_past[:num_train]
        val_spaced = spaced_past[num_train : num_train + num_val]
        test_spaced = spaced_past[num_train + num_val :]
    else:
        train_spaced = val_spaced = test_spaced = None

    train_future = future_inputs[:num_train]
    val_future = future_inputs[num_train : num_train + num_val]
    test_future = future_inputs[num_train + num_val :]

    train_targets = outputs[:num_train]
    val_targets = outputs[num_train : num_train + num_val]
    test_targets = outputs[num_train + num_val :]

    return (
        train_dense,
        train_spaced,
        train_future,
        train_targets,
        val_dense,
        val_spaced,
        val_future,
        val_targets,
        test_dense,
        test_spaced,
        test_future,
        test_targets,
    )


def prepare_data(
    file_path,
    dense_lookback,
    spaced_lookback,
    forecast_horizon,
    future_cols,
    target_col,
    spaced=True,
    step_growth_factor=None,
    pca=True
    ):
    """
    Prepare data for forked training with dense and spaced lookback.
    Scales all features except sine/cosine temporal columns and explicitly dropped columns.
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        dense_lookback (int): Number of consecutive timesteps for dense lookback.
        spaced_lookback (int): Maximum lookback period for spaced lookback.
        forecast_horizon (int): Number of timesteps to forecast.
        future_cols (list): Names of future columns.
        target_col (str): Name of the target column.
        spaced (bool): Whether to include spaced lookback features.
        step_growth_factor (float): Factor for dynamic spacing of spaced lookback.
    Returns:
        Scaled datasets (train, val, test) and scalers.
    """
    # Load and preprocess data
    data = pd.read_csv(file_path)
    data = data.drop(columns=["Year", "Month", "Day", "Hour", "Year_Scaled", "Volume_MWh", "Diff"])
    data = data.dropna().reset_index(drop=True)

    # Identify columns to exclude from scaling
    exclude_from_scaling = []
    # exclude_from_scaling = ["Hour_Sin", "Hour_Cos", "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos"]


    # Separate features and target
    output_data = np.array(data.loc[:, target_col]).reshape(-1, 1)
    future_indices = [data.columns.get_loc(col) for col in future_cols]
    feature_cols = [col for col in data.columns if col not in exclude_from_scaling and col != target_col]

    # Prepare datasets
    (
        train_dense_past,
        train_spaced_past,
        train_future,
        train_targets,
        val_dense_past,
        val_spaced_past,
        val_future,
        val_targets,
        test_dense_past,
        test_spaced_past,
        test_future,
        test_targets,
    ) = create_datasets(
        features=data.to_numpy(),
        target=output_data,
        dense_lookback=dense_lookback,
        spaced_lookback=spaced_lookback,
        forecast_horizon=forecast_horizon,
        future_cols=future_indices,
        p_train=0.7,
        p_val=0.15,
        p_test=0.15,
        spaced=spaced,
        step_growth_factor=step_growth_factor,
    )

    # Initialize scalers for features
    scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in feature_cols}

    # Apply scaling
    for col in feature_cols:
        col_idx = data.columns.get_loc(col)

        # Fit scaler on training dense past data
        flat_train_dense = train_dense_past[:, :, col_idx].flatten().reshape(-1, 1)
        scalers[col].fit(flat_train_dense)

        # Transform dense past data
        for dataset in [train_dense_past, val_dense_past, test_dense_past]:
            dataset[:, :, col_idx] = scalers[col].transform(
                dataset[:, :, col_idx].reshape(-1, 1)
            ).reshape(dataset.shape[0], dataset.shape[1])

        # If spaced lookback exists, transform spaced past data
        if spaced:
            past_scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in feature_cols}
            flat_train_spaced = train_spaced_past[:, :, col_idx].flatten().reshape(-1, 1)
            past_scalers[col].fit(flat_train_spaced)
            for dataset in [train_spaced_past, val_spaced_past, test_spaced_past]:
                dataset[:, :, col_idx] = past_scalers[col].transform(
                    dataset[:, :, col_idx].reshape(-1, 1)
                ).reshape(dataset.shape[0], dataset.shape[1])

    # Scale the target column
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).reshape(train_targets.shape)
    val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).reshape(val_targets.shape)
    test_targets = target_scaler.transform(test_targets.reshape(-1, 1)).reshape(test_targets.shape)

    print(train_targets)
    print(val_targets)

    # Apply PCA 
    if pca:
        # Reshape dense past data
        n_timesteps, n_features = train_dense_past.shape[1], train_dense_past.shape[2]
        train_dense_reshaped = train_dense_past.reshape(-1, n_features)
        val_dense_reshaped = val_dense_past.reshape(-1, n_features)
        test_dense_reshaped = test_dense_past.reshape(-1, n_features)

        # Apply PCA 
        pca_model = PCA()
        pca_model.fit(train_dense_reshaped)

        explained_variance_ratio = pca_model.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Select components that explain at least 1% of variance
        components_above_threshold = np.where(explained_variance_ratio >= 0.01)[0]
        optimal_components = len(components_above_threshold)

        print(f"Number of components explaining ≥1% variance: {optimal_components}")
        print(f"Total variance explained: {cumulative_explained_variance[optimal_components-1]:.4f}")

        pca_model = PCA(n_components=optimal_components)
        pca_model.fit(train_dense_reshaped)  # Fit PCA on the training data

        # Transform datasets
        train_dense_pca = pca_model.transform(train_dense_reshaped).reshape(
            train_dense_past.shape[0], n_timesteps, -1
        )
        val_dense_pca = pca_model.transform(val_dense_reshaped).reshape(
            val_dense_past.shape[0], n_timesteps, -1
        )
        test_dense_pca = pca_model.transform(test_dense_reshaped).reshape(
            test_dense_past.shape[0], n_timesteps, -1
        )

        # Replace dense past datasets
        train_dense_past = torch.tensor(train_dense_pca, dtype=torch.float32)
        val_dense_past = torch.tensor(val_dense_pca, dtype=torch.float32)
        test_dense_past = torch.tensor(test_dense_pca, dtype=torch.float32)


    # Convert datasets to PyTorch tensors
    train_dense_past = torch.tensor(train_dense_past, dtype=torch.float32)
    val_dense_past = torch.tensor(val_dense_past, dtype=torch.float32)
    test_dense_past = torch.tensor(test_dense_past, dtype=torch.float32)
    train_future = torch.tensor(train_future, dtype=torch.float32)
    val_future = torch.tensor(val_future, dtype=torch.float32)
    test_future = torch.tensor(test_future, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    if spaced:
        train_spaced_past = torch.tensor(train_spaced_past, dtype=torch.float32)
        val_spaced_past = torch.tensor(val_spaced_past, dtype=torch.float32)
        test_spaced_past = torch.tensor(test_spaced_past, dtype=torch.float32)
        return (
            train_dense_past,
            train_spaced_past,
            train_future,
            train_targets,
            val_dense_past,
            val_spaced_past,
            val_future,
            val_targets,
            test_dense_past,
            test_spaced_past,
            test_future,
            test_targets,
            target_scaler,
        )
    else:
        return (
            train_dense_past,
            train_future,
            train_targets,
            val_dense_past,
            val_future,
            val_targets,
            test_dense_past,
            test_future,
            test_targets,
            target_scaler,
        )