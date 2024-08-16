import yfinance as yf
import pandas as pd
import os

# Define the ticker symbol for Bitcoin
ticker_symbol = 'BTC-USD'

# Define the start and end dates
start_date = '2021-01-01'
end_date = '2024-12-31'

# Download the hourly price data
btc_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

# Define the folder name and create it if it doesn't exist
folder_name = 'datasets'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define the file name and path
file_name = 'Bitcoin_Daily_from_2021.csv'
file_path = os.path.join(folder_name, file_name)

# Save the data to the CSV file in the 'datasets' folder
btc_data.to_csv(file_path)

# Display the first few rows of the data
print(btc_data.head())

print(f"Data saved to {file_path}")
