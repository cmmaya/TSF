import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta

# Load Data
def load_data(file_path, close_col):
    data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    data = data.dropna()
    return data[close_col]

# Transform data for LSTM
def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

# Plot the original and predicted data
def plot_predictions(original_data, predictions, n_train_days, num_past_days, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(original_data.index[num_past_days:], original_data.values[num_past_days:], label='Target')
    plt.plot(original_data.index[num_past_days:], list(predictions), label='Predicted')
    plt.axvline(x=original_data.index[n_train_days], label='Prediction Start', ymin=0.1, ymax=0.75, linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# LSTM model definition
def build_and_train_lstm(train_X, train_y, num_past_days, num_features, epochs=20, batch_size=5):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, shuffle=False)
    return model, history

# Predict future prices
def predict_future(model, last_sequence, num_days, num_features):
    predictions = []
    current_sequence = last_sequence
    for _ in range(num_days):
        next_prediction = model.predict(current_sequence).flatten()[0]
        predictions.append(next_prediction)
        next_sequence = np.roll(current_sequence, -1, axis=1)
        next_sequence[:, -1, :] = next_prediction
    return predictions

def main():
    folder_name = 'datasets'
    file_name = 'Bitcoin_Daily_from_2021.csv'
    file_path = os.path.join(folder_name, file_name)
    close_col = 'Adj_Close'
    Output_address = "C:\\Users\\camil\\Documents\\Juanp\\TSF\\results\\"

    # Load and preprocess data
    original_data = load_data(file_path, close_col)
    num_past_days = 20
    num_future_days = 180  # 6 months prediction
    values = original_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    supervised_data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
    values = supervised_data.values

    # Split data into training and test sets
    n_train_days = int(0.9 * len(supervised_data))
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    num_features = 1  # since we're only using the 'Adj_Close' column

    train_X, train_y = train[:, :num_past_days * num_features], train[:, -num_features]
    test_X, test_y = test[:, :num_past_days * num_features], test[:, -num_features]
    train_X = train_X.reshape((train_X.shape[0], num_past_days, num_features))
    test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))

    # Build and train the model
    model, history = build_and_train_lstm(train_X, train_y, num_past_days, num_features)

    # Predictions for test set
    yhat_test = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], num_past_days * num_features))

    inv_yhat_test = scaler.inverse_transform(np.concatenate((yhat_test, test_X[:, -(num_features-1):]), axis=1))
    inv_yhat_test = inv_yhat_test[:, 0]

    # Predict future prices
    last_sequence = scaled[-num_past_days:].reshape((1, num_past_days, num_features))
    future_predictions = predict_future(model, last_sequence, num_future_days, num_features)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Concatenate last 6 months of actual data with predictions
    last_6_months_actual = original_data[-180:]  # last 180 days (approx. 6 months)
    future_dates = [last_6_months_actual.index[-1] + timedelta(days=i) for i in range(1, num_future_days + 1)]
    future_data = pd.Series(future_predictions, index=future_dates)

    concatenated_data = pd.concat([last_6_months_actual, future_data])

    # Plot the result
    plt.figure(figsize=(10, 5))
    plt.plot(concatenated_data.index, concatenated_data.values, label='Predicted')
    plt.axvline(x=last_6_months_actual.index[-1], label='Prediction Start', ymin=0.1, ymax=0.75, linestyle='--')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('BTC/USD')
    plt.title('Bitcoin Price Prediction for the Next 6 Months')
    plt.show()

if __name__ == "__main__":
    main()
