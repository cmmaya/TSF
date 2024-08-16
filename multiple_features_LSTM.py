import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import os

folder_name = 'datasets'
file_name = 'Bitcoin_Daily_from_2021.csv'
file_path = os.path.join(folder_name, file_name)
close = 'Adj_Close'
Output_address = "C:\\Users\\camil\\Documents\\Juanp\\TSF\\results\\"

# Basically loading the data and making a data-frame wrt to time.
def data_loader():
   cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
   data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
   data.columns = cols
   data = data.dropna()
   print(f"The Shape of the Data-Set is : {data.shape}\nThe Data-Set is : \n{data.head()}\n")
   return data

def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    return agg

def normalize_values(values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaler, scaled

def split_test_train(scaled, data, num_past_days):
    num_features = len(data.columns)
    data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
    values = data.values
    num_obs = num_features * num_past_days

    # Split into training and test set
    n_train_days = int(0.9 * len(data))
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    train_X, train_y = train[:, :num_obs], train[:, -num_features]
    test_X, test_y = test[:, :num_obs], test[:, -num_features]

    # Reshape to fit data into the model
    train_X = train_X.reshape((train_X.shape[0], num_past_days, num_features))
    test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))
    return train_X, train_y, test_X, test_y

def setup_model(scaled, data, num_past_days: int, n: int, batch_size, epochs):
    train_X, train_y, test_X, test_y = split_test_train(scaled, data, num_past_days)
    # Setup the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(n, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')

    # Train the model
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=1, shuffle=False)
    return model, history

def main():
    data = data_loader()
    original_data = data
    values = data.values
    scaler, scaled = normalize_values(values)
    num_past_days = 20
    num_features = len(data.columns)
    n_train_days = int(0.9 * len(data))

    model, history = setup_model(scaled, data, num_past_days, 100, 5, 10)

    # Predictions
    train_X, _, test_X, _ = split_test_train(scaled, data, num_past_days)
    yhat_test = model.predict(test_X)
    yhat_train = model.predict(train_X)
    test_X = test_X.reshape((test_X.shape[0], num_past_days * num_features))
    train_X = train_X.reshape((train_X.shape[0], num_past_days * num_features))

    # Invert scaling for forecast
    inv_yhat_test = np.concatenate((yhat_test, test_X[:, -(num_features-1):]), axis=1)
    inv_yhat_train = np.concatenate((yhat_train, train_X[:, -(num_features-1):]), axis=1)
    inv_yhat_test = scaler.inverse_transform(inv_yhat_test)
    inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
    inv_yhat_test = inv_yhat_test[:, 0]
    inv_yhat_train = inv_yhat_train[:, 0]

    # Plot the loss function
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(original_data.index[n_train_days:], original_data['Adj_Close'][n_train_days:], label='Target')
    plt.plot(original_data.index[n_train_days + num_past_days:], list(inv_yhat_test), label='Predicted')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('BTC/USD')
    plt.show()

main()