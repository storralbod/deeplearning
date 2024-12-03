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


