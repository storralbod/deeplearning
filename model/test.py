import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load the updated merged data
data = pd.read_csv('processed_marginal_data.csv')

# Define input and output sequence lengths
sequence_length = 336  # Use 2 weeks of history for predictions
output_sequence_length = 24  # Predict the next 24 hours

# Prepare the input features and target prices
X, y = [], []

for i in range(len(data) - sequence_length - output_sequence_length):
    features = data[['Price1', 'Day_Sin', 'Day_Cos', 'Month_Sin', 'Month_Cos', 'Year']].values
    X.append(features[i:i + sequence_length])  # input sequence
    y.append(features[i + sequence_length:i + sequence_length + output_sequence_length, 0])  # output sequence of prices

X = np.array(X)  # Shape: (samples, sequence_length, features)
y = np.array(y)  # Shape: (samples, output_sequence_length)

# Normalize data
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_shape = X.shape
X = scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X_shape)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y = scaler_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Create TensorFlow datasets for training and testing
batch_size = 32

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build the LSTM model
model = Sequential([
    LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(units=output_sequence_length)  # Output layer for sequence prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs = 50  # Adjust based on your needs

model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

# Evaluate the model
train_loss = model.evaluate(train_dataset, verbose=0)
test_loss = model.evaluate(test_dataset, verbose=0)
print(f'Train Loss: {train_loss:.6f}')
print(f'Test Loss: {test_loss:.6f}')
