import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error
from math import sqrt

# Define the folder and file name
folder_name = 'datasets'
file_name = 'Bitcoin_Price_History_2023_to_2024.csv'
file_path = os.path.join(folder_name, file_name)

# Load the dataset
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Use the first 400 dates
data = data.head(400)

# Extract the 'Adj Close' column
prices = data['Adj Close']

# Split the data into training and testing sets
train_size = int(len(prices) * 0.8)  # 80% training data
train, test = prices[:train_size], prices[train_size:]

# Define the ARIMA model
model = ARIMA(train, order=(5, 1, 0))  # (p, d, q) parameters

# Fit the model
model_fit = model.fit()

# Make predictions
forecast = model_fit.forecast(steps=len(test))

# Compare predictions to actual values
test_predictions = forecast
test_actuals = test

# Calculate accuracy metrics
mae = mean_absolute_error(test_actuals, test_predictions)
mse = mean_squared_error(test_actuals, test_predictions)
rmse = sqrt(mse)

# Print results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
