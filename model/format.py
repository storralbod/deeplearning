import pandas as pd
import numpy as np
import glob

# File paths for the marginalpdbc files
marginalpdbc_files = glob.glob('electricproject/deeplearning/data_23/marginalpdbc*.csv')
print(f"Found {len(marginalpdbc_files)} marginalpdbc files")

# Define a function to encode hours in sine and cosine format
def encode_time(hour):
    max_hour = 24
    hour_sin = np.sin(2 * np.pi * hour / max_hour)
    hour_cos = np.cos(2 * np.pi * hour / max_hour)
    return hour_sin, hour_cos

# Initialize an empty list to store data from each file
marginal_data = []

# Process each marginalpdbc file
for file in marginalpdbc_files:
    print(f"Processing file: {file}")
    # Read only the first 6 columns, skipping the first row and removing the last row
    data = pd.read_csv(file, delimiter=';', header=None, skiprows=1, usecols=range(6), encoding='latin1').iloc[:-1, :]
    # Rename columns based on the format provided
    data.columns = ['Year', 'Month', 'Day', 'Hour', 'Price1', 'Price2']
    # Select relevant columns and drop rows with missing values
    data = data[['Year', 'Month', 'Day', 'Hour', 'Price1']].dropna()
    
    # Convert the 'Hour' column to numeric
    data['Hour'] = pd.to_numeric(data['Hour'], errors='coerce')
    
    # Filter out rows with invalid or missing hour values
    data = data.dropna(subset=['Hour'])
    
    # Encode the hour as sine and cosine values
    data['Hour_Sin'], data['Hour_Cos'] = zip(*data['Hour'].apply(encode_time))
    
    # Append the processed data to the list
    marginal_data.append(data)

# Concatenate all daily data into a single DataFrame
merged_data = pd.concat(marginal_data, ignore_index=True)

# Save the merged data to a single CSV file
merged_data.to_csv('merged_marginal_data.csv', index=False)
print("Data has been merged and saved to 'merged_marginal_data.csv'")
