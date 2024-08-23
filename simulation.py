import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tqdm import tqdm
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Data
folder_name = 'datasets'
file_name = 'Eth_Daily_from_2023_S2.csv'
file_path = os.path.join(folder_name, file_name)

cols = ["Open", "High", "Low", "Close", "Volume"]
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
data = data[cols].dropna()
num_features = len(data.columns)

# Convert prices to returns
returns = data['Close'].pct_change().dropna()

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
powe
# Convert to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{j+1}(t-{i})') for j in range(data.shape[1])]
    cols.append(df)
    names += [(f'{j+1}(t)') for j in range(data.shape[1])]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

# Simulation Function
def run_simulation(returns, prices, scaler, n_past_days=12, thresh=0.01, amt=10000, plot=True):
    curr_holding = False
    events_list = []
    init_amt = amt

    for i in tqdm(range(n_past_days, len(returns))):
        # Prepare data up to current date
        current_data = prices.iloc[:i+1]
        num_obs = num_features * n_past_days

        # Skip if not enough data to form the required sequence
        if len(current_data) < n_past_days + 1:
            continue

        # Convert current data to supervised learning format
        scaled_current_data = scaler.transform(current_data)
        supervised_data = series_to_supervised(scaled_current_data, n_in=n_past_days)
        
        train_X = supervised_data.values[:, :num_obs]        
        train_y = supervised_data.values[:, -num_features:]       


        train_X = train_X.reshape((train_X.shape[0], n_past_days, data.shape[1]))

        # Train the LSTM Model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(tf.keras.layers.Dense(1))  # Predicting only the "Close" price
        model.compile(loss='mae', optimizer='adam')

        model.fit(train_X, train_y, epochs=20, batch_size=5, verbose=0, shuffle=False)

        # If holding the stock, sell it
        if curr_holding:
            sell_price = prices.iloc[i]['Close']
            curr_holding = False
            ret = (sell_price - buy_price) / buy_price
            amt *= (1 + ret)
            events_list.append(('s', prices.index[i], ret))
            continue

        # Prepare input data for prediction
        last_sequence = scaled_current_data[-n_past_days:]
        last_sequence = last_sequence.reshape((1, n_past_days, len(cols)))

        # Predict the return
        pred_close = model.predict(last_sequence)[0][0]
        pred_close = scaler.inverse_transform([[0, 0, 0, pred_close, 0]])[0][3]
        pred_return = (pred_close - prices.iloc[i-1]['Close']) / prices.iloc[i-1]['Close']

        # Buy if prediction exceeds threshold
        if pred_return > thresh:
            curr_holding = True
            buy_price = prices.iloc[i]['Close']
            events_list.append(('b', prices.index[i]))

    # Plot the results
    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(prices['Close'], label='Actual Close Price')
        for idx, event in enumerate(events_list):
            plt.axvline(event[1], color='k', linestyle='--', alpha=0.4)
            if event[0] == 's':
                color = 'green' if event[2] > 0 else 'red'
                plt.fill_betweenx([prices['Close'].min(), prices['Close'].max()], 
                                  event[1], events_list[idx-1][1], color=color, alpha=0.1)

        tot_return = round(100 * (amt / init_amt - 1), 2)
        plt.title(f"Stock Price Data\nThreshold={thresh}\nTotal Amount: ${round(amt,2)}\nTotal Return: {tot_return}%", fontsize=20)
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.show()

    return amt

# Run the simulation
final_amount = run_simulation(returns, data, scaler, thresh=0.01)
print(f"Final amount after simulation: ${final_amount:.2f}")