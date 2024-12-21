# Multi-Horizon Probability Forecaster for Electricity Market Arbitrage
This project explores deep learning techniques for electricity price forecasting to enhance strategic bidding in day-ahead and intraday markets.

## Repository Structure

### **Key Files and Directories**

- **`data_processing/`**: Contains scripts and notebooks for preparing and formatting raw data.
  - `Format.ipynb`: Core notebook for preparing raw data for training. This includes:
    - Reading raw data files.
    - Combining data into a unified dataset.
    - Forward-filling missing values.
    - Merging data on `datetime` to ensure compatibility.
    - Produces the final formatted data for training.
  - Other files: Supporting scripts for specific data processing tasks (e.g., weather data).

- **`formatted_data/`**: Directory containing data files that are formatted and ready for creating datasets.

- **`model_functions/`**: Contains key scripts used during model development and training.
  - `data_prep.py`: Functions for dataset preparation, including:
    - Min-Max scaling.
    - Dataset creation utilities.
  - `model.py`: Implements various model architectures for forecasting.
  - Notebooks documenting the full training and validation process:
    - `Best_Version_LSTM_DA_santi.ipynb`: Comprehensive training and validation process for LSTM-MLPs.
    - `notebook.ipynb`: Experimental notebook covering multiple stages of model training and testing.
    - `plotting.ipynb`: Visualization of model outputs and performance.
    - `testtamas.ipynb`: Additional tests and experimental workflows.

- **`raw_data/`**: Directory containing the raw data files. These are the original datasets used as input for the data processing pipeline.


