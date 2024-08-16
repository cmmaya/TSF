# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import os

# Register Matplotlib converters
register_matplotlib_converters()

# Define file paths and column names
folder_name = 'datasets'
file_name = 'Bitcoin_Daily_from_2021.csv'
file_path = os.path.join(folder_name, file_name)
Output_address = "C:\\Users\\camil\\Documents\\Juanp\\TSF\\results\\"
cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]

# Data Loading Function
def data_loader():
    data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    data.columns = cols
    data = data.dropna()
    print(f"The Shape of the Data-Set is : {data.shape}\nThe Data-Set is : \n{data.head()}\n")
    return data

# Convert time series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]
    
    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]
    
    # Concatenate all columns and drop NaN values
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    return agg

# Prepare data for model training
def prepare_data(data, num_past_days):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.values)
    supervised_data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
    num_features = len(data.columns)
    values = supervised_data.values
    num_obs = num_features * num_past_days

    # Split into training and test set
    n_train_days = int(0.9 * len(supervised_data))
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    train_X, train_y = train[:, :num_obs], train[:, -num_features]
    test_X, test_y = test[:, :num_obs], test[:, -num_features]

    train_X = train_X.reshape((train_X.shape[0], num_past_days, num_features))
    test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))

    return train_X, train_y, test_X, test_y, scaler, supervised_data.index[n_train_days:]

# Build and train the LSTM model
def train_model(train_X, train_y, test_X, test_y, num_past_days, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(num_past_days, num_features)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=20, batch_size=5, validation_data=(test_X, test_y), verbose=1, shuffle=False)
    return model, history

# Invert scaling and make predictions
def make_predictions(model, train_X, test_X, train_y, test_y, scaler, num_past_days, num_features):
    yhat_train = model.predict(train_X)
    yhat_test = model.predict(test_X)

    # Reshape to original format
    train_X = train_X.reshape((train_X.shape[0], num_past_days * num_features))
    test_X = test_X.reshape((test_X.shape[0], num_past_days * num_features))

    # Invert scaling for predictions
    inv_yhat_train = np.concatenate((yhat_train, train_X[:, -(num_features-1):]), axis=1)
    inv_yhat_test = np.concatenate((yhat_test, test_X[:, -(num_features-1):]), axis=1)
    inv_yhat_train = scaler.inverse_transform(inv_yhat_train)[:, 0]
    inv_yhat_test = scaler.inverse_transform(inv_yhat_test)[:, 0]

    return inv_yhat_train, inv_yhat_test

# Plot the loss function
def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the predictions
def plot_predictions(original_data, inv_yhat_train, inv_yhat_test, num_past_days, n_train_days):
    plt.plot(original_data.index[num_past_days:], original_data['Adj_Close'][num_past_days:], label='Target')
    plt.plot(original_data.index[num_past_days:], list(inv_yhat_train) + list(inv_yhat_test), label='Predicted')
    plt.axvline(x=original_data.index[n_train_days], label='Prediction Start', ymin=0.1, ymax=0.75, linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('BTC/USD')
    plt.show()

    plt.plot(original_data.index[n_train_days:], original_data['Adj_Close'][n_train_days:], label='Target')
    plt.plot(original_data.index[n_train_days + num_past_days:], list(inv_yhat_test), label='Predicted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('BTC/USD')
    plt.show()

# Main function to run the entire pipeline
def main():
    num_past_days = 20
    data = data_loader()
    train_X, train_y, test_X, test_y, scaler, n_train_days = prepare_data(data, num_past_days)
    model, history = train_model(train_X, train_y, test_X, test_y, num_past_days, len(data.columns))
    inv_yhat_train, inv_yhat_test = make_predictions(model, train_X, test_X, train_y, test_y, scaler, num_past_days, len(data.columns))
    plot_loss(history)
    plot_predictions(data, inv_yhat_train, inv_yhat_test, num_past_days, n_train_days)


main()
