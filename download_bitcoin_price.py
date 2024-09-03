import yfinance as yf
import pandas as pd
import os
import requests
from datetime import datetime

def download_yahoo_symbol_metrics(ticker_symbol, start_date, end_date):

    # Download the hourly price data
    btc_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1d')

    # Define the folder name and create it if it doesn't exist
    folder_name = 'datasets'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Define the file name and path
    file_name = 'Eth_Daily_from_2023_S2.csv'
    file_path = os.path.join(folder_name, file_name)

    # Save the data to the CSV file in the 'datasets' folder
    btc_data.to_csv(file_path)

    # Display the first few rows of the data
    print(btc_data.head())

    print(f"Data saved to {file_path}")

import requests
import pandas as pd

def get_binance_orderbook(symbol='BTCUSDT', limit=100):
    """
    Fetches the order book data for a specified symbol from Binance.
    
    :param symbol: The trading pair symbol (e.g., 'BTCUSDT' for Bitcoin to USDT).
    :param limit: The number of orders to retrieve (can be 5, 10, 20, 50, 100, 500, or 1000).
    :return: A tuple containing two pandas DataFrames: bids and asks.
    """
    base_url = 'https://api.binance.com/api/v3/depth'
    
    # Define parameters for the API request
    params = {
        'symbol': symbol,
        'limit': limit
    }
    
    # Send a GET request to the Binance API
    response = requests.get(base_url, params=params)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from Binance API. Status code: {response.status_code}, Response: {response.text}")
    
    # Parse the JSON response
    data = response.json()
    
    # Convert the bids and asks to pandas DataFrames
    bids = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
    asks = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
    
    # Convert price and quantity columns to float
    bids['price'] = bids['price'].astype(float)
    bids['quantity'] = bids['quantity'].astype(float)
    asks['price'] = asks['price'].astype(float)
    asks['quantity'] = asks['quantity'].astype(float)


    # Get the order book data
    bids, asks = get_binance_orderbook(symbol=symbol, limit=limit)
    
    # Define filenames
    bids_filename = f'{symbol}_bids.csv'
    asks_filename = f'{symbol}_asks.csv'
    
    # Save to CSV files
    bids.to_csv(bids_filename, index=False)
    asks.to_csv(asks_filename, index=False)
    
    print(f"Order book data saved to {bids_filename} and {asks_filename}.")

# Example usage:
get_binance_orderbook(symbol='BTCUSDT', limit=100)


# Define the ticker symbol for Bitcoin
ticker_symbol = 'ETH-USD'

# Define the start and end dates
start_date = '2023-06-01'
end_date = '2023-12-31'

# download_yahoo_symbol_metrics(ticker_symbol, start_date, end_date)