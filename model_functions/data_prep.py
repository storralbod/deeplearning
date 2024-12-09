import pandas as pd 
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset
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

class Dataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        # Return the size of the dataset
        return len(self.targets)

    def __getitem__(self, index):
        # Retrieve inputs and targets at the given index
        X = self.inputs[index]
        y = self.targets[index]

        return X, y


def create_datasets(input_data,output_data, lag_in_days, forecast_horizon_in_hours, dataset_class, p_train=0.9, p_val=0.1, p_test=0):

    assert len(input_data) == len(output_data)

    hours = len(input_data)
    lag_in_hours = lag_in_days*24
    #forecast_hours = forecast_horizon_in_days*24

    num_train = int((hours-lag_in_hours-forecast_horizon_in_hours)/forecast_horizon_in_hours*p_train)
    num_val = int((hours-lag_in_hours-forecast_horizon_in_hours)/forecast_horizon_in_hours*p_val)
    num_test = int((hours-lag_in_hours-forecast_horizon_in_hours)/forecast_horizon_in_hours*p_test)


    # Creating features and labels dataset
    def create_features(input_data, output_data, lag_in_hours, hours):

        # features dataset to include prices from [t:t-14, t-1year-14:t-1year+14]. So from present to 2 weeks in past and what ocurred one_year before two weeks behind and ahead
        # for now (small dataset) prices from [t:t-14]
        inputs, outputs = [], []
        for i in range(12,hours,24): # features only given for clearing time t=12h everyday

            if i+lag_in_hours+forecast_horizon_in_hours > hours:
                break

            inputs.append(input_data[i:i+lag_in_hours,:])
            outputs.append(output_data[i+lag_in_hours:i+lag_in_hours+forecast_horizon_in_hours])

        return np.array(inputs), np.array(outputs)

    inputs, outputs = create_features(input_data,output_data, lag_in_hours, hours)
    training_set = dataset_class(inputs[:num_train],outputs[:num_train])
    val_set = dataset_class(inputs[num_train:num_train+num_val],outputs[num_train:num_train+num_val])
    test_set = dataset_class(inputs[-num_test:], outputs[-num_test:])

    return training_set, val_set, test_set