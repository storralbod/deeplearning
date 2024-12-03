# %%
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


# %%

data_files_path = glob.glob(r"C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 2\Semester 1\Deep Learning\Project\deeplearning\data_23"+'/*.csv')

# %%



# %%
# Changing str columns to floats
str_columns_da = [column for column in data_da.columns if type(data_da[column].iloc[0])==str]
str_columns_id = [column for column in data_id.columns if type(data_id[column].iloc[0])==str]

for column in str_columns_da:
    data_da[column] = data_da[column].astype(float)
for column in str_columns_id:
    data_id[column] = data_id[column].astype(float)

# checking all columns are floats:
print([data_da[column].iloc[0] for column in data_da.columns])
print([data_id[column].iloc[0] for column in data_id.columns])

# saving both DA and ID data in csv files
data_da.to_csv(r"C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 2\Semester 1\Deep Learning\Project\deeplearning\models_santi\data_santi\data_da.csv", index=False)
data_id.to_csv(r"C:\Users\storr\OneDrive - Danmarks Tekniske Universitet\Year 2\Semester 1\Deep Learning\Project\deeplearning\models_santi\data_santi\data_id.csv", index=False)


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



class Dataset(data.Dataset):
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

