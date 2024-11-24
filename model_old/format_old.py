import pandas as pd
import numpy as np
import glob
import os

# File paths for the marginalpdbc files, weather data, and gas price directory
marginalpdbc_files = glob.glob('/home/teitur/DTU/electricproject/deeplearning/data_teitur/marginalpdbc_*.1')
weather_data_file = '/home/teitur/DTU/electricproject/deeplearning/model/spain_weather_data_2018_2023.csv'
gas_price_dir = '/home/teitur/DTU/electricproject/deeplearning/gas/output/csv/'

print(f"Found {len(marginalpdbc_files)} marginalpdbc files")

# Function to encode time (day, month, hour) as sine and cosine
def encode_time(value, max_value):
    value_sin = np.sin(2 * np.pi * value / max_value)
    value_cos = np.cos(2 * np.pi * value / max_value)
    return value_sin, value_cos

# Load and preprocess weather data
print(f"Loading weather data from {weather_data_file}...")
weather_data = pd.read_csv(weather_data_file)

if 'time' in weather_data.columns:
    weather_data['time'] = pd.to_datetime(weather_data['time'])
    weather_data['Year'] = weather_data['time'].dt.year
    weather_data['Month'] = weather_data['time'].dt.month
    weather_data['Day'] = weather_data['time'].dt.day
    weather_data['Hour'] = weather_data['time'].dt.hour

# Retain relevant columns and aggregate across cities
weather_columns = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'cloudcover']
weather_data = weather_data[['Year', 'Month', 'Day', 'Hour'] + weather_columns]
weather_data = weather_data.groupby(['Year', 'Month', 'Day', 'Hour']).mean().reset_index()

print(f"Aggregated weather data rows: {len(weather_data)}")
print(weather_data.head())

# Load and preprocess all gas price data
print(f"Processing gas price files in directory: {gas_price_dir}")
gas_files = glob.glob(os.path.join(gas_price_dir, 'MIBGAS_Data_*.csv'))
all_gas_data = []

for gas_file in gas_files:
    print(f"Processing file: {gas_file}")
    try:
        gas_data = pd.read_csv(gas_file)
        gas_data.columns = gas_data.columns.str.strip().str.replace('\n', ' ', regex=False)  # Normalize column names
        gas_data = gas_data[['Trading Day', 'Price [EUR/MWh]']].rename(columns={
            'Trading Day': 'Trading_Day',
            'Price [EUR/MWh]': 'Gas_Price'
        })
        gas_data['Trading_Day'] = pd.to_datetime(gas_data['Trading_Day'])
        gas_data['Year'] = gas_data['Trading_Day'].dt.year
        gas_data['Month'] = gas_data['Trading_Day'].dt.month
        gas_data['Day'] = gas_data['Trading_Day'].dt.day
        all_gas_data.append(gas_data[['Year', 'Month', 'Day', 'Gas_Price']])
    except Exception as e:
        print(f"Error processing file {gas_file}: {e}")

# Combine all gas price data
gas_data = pd.concat(all_gas_data, ignore_index=True)

# Forward-fill missing Gas_Price values
gas_data = gas_data.sort_values(by=['Year', 'Month', 'Day'])
gas_data['Gas_Price'] = gas_data['Gas_Price'].fillna(method='ffill')
gas_data['Gas_Price'] = gas_data['Gas_Price'].fillna(method='bfill')  # Backfill if forward-fill doesn't work

print(f"Combined gas data rows after filling: {len(gas_data)}")
print(gas_data.head())

# Process marginalpdbc files and merge all data
marginal_data = []

for file in marginalpdbc_files:
    print(f"Processing file: {file}")
    data = pd.read_csv(file, delimiter=';', header=None, skiprows=1, usecols=range(6), encoding='latin1').iloc[:-1, :]
    data.columns = ['Year', 'Month', 'Day', 'Hour', 'Price1', 'Price2']
    data = data[['Year', 'Month', 'Day', 'Hour', 'Price1']].dropna()
    data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
    data['Month'] = pd.to_numeric(data['Month'], errors='coerce')
    data['Day'] = pd.to_numeric(data['Day'], errors='coerce')
    data = data.dropna(subset=['Hour', 'Month', 'Day'])
    
    # Encode time features
    data['Hour_Sin'], data['Hour_Cos'] = zip(*data['Hour'].apply(lambda x: encode_time(x, 24)))
    data['Day_Sin'], data['Day_Cos'] = zip(*data['Day'].apply(lambda x: encode_time(x, 31)))
    data['Month_Sin'], data['Month_Cos'] = zip(*data['Month'].apply(lambda x: encode_time(x, 12)))

    # Ensure Year is numeric and scaled
    data['Year'] = data['Year'].astype(int)
    data['Year_Scaled'] = (data['Year'] - 2018) * 0.1 + 0.1

    # Merge with weather data
    merged_data = pd.merge(data, weather_data, on=['Year', 'Month', 'Day', 'Hour'], how='inner')
    # Merge with gas price data
    merged_data = pd.merge(merged_data, gas_data, on=['Year', 'Month', 'Day'], how='left')

    marginal_data.append(merged_data)

# Combine all marginal data
marginal_data = pd.concat(marginal_data, ignore_index=True)

# Select desired columns and save to a CSV file
columns_to_keep = [
    'Price1', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 
    'Month_Sin', 'Month_Cos', 'Year_Scaled', 'temperature_2m', 
    'precipitation', 'wind_speed_10m', 'cloudcover', 'Gas_Price'
]
marginal_data = marginal_data[columns_to_keep]

output_file = '/home/teitur/DTU/electricproject/deeplearning/model/data_weather_gas.csv'
marginal_data.to_csv(output_file, index=False)
print(f"Data has been processed and saved to '{output_file}'")

print(marginal_data.head())
