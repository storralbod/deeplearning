import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define hyperparameters as variables
hyperparameters = {
    "units": 50,
    "dropout": 0.2,
    "sequence_length": 336,
    "output_sequence_length": 24,
    "feature_count": 8,  # Number of input features
    "learning_rate": 0.001,
    "batch_size": 32,
}

# Define model functions
def multi_output_lstm(hyperparams):
    model = Sequential([
        LSTM(units=hyperparams["units"], input_shape=(hyperparams["sequence_length"], hyperparams["feature_count"])),
        Dense(units=hyperparams["output_sequence_length"])  # Predict the entire sequence at once
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def single_output_lstm(hyperparams):
    model = Sequential([
        LSTM(units=hyperparams["units"], return_sequences=True, input_shape=(hyperparams["sequence_length"], hyperparams["feature_count"])),
        Dropout(rate=hyperparams["dropout"]),
        Dense(units=1)  # Predict one value at a time
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Store models in a dictionary
model_dict = {
    "multi_output": multi_output_lstm,
    "single_output": single_output_lstm,
}
