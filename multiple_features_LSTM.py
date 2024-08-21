import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Load Data
folder_name = 'datasets'
file_name = 'Eth_Daily_from_2018.csv'
file_path = os.path.join(folder_name, file_name)

close = 'Adj_Close'
Output_address = "C:\\Users\\camil\\Documents\\Juanp\\TSF\\results\\"
cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
data.columns = cols
data = data.dropna()
print(f"The Shape of the Data-Set is : {data.shape}\nThe Data-Set is : \n{data.head()}\n")

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

def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse,rmse,mae

## FIRST: we organize the data for correct input
current_datetime = datetime.now().date().isoformat()
original_data = data
values = data.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

num_features = len(data.columns)
num_past_days = 30
data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
values = data.values
num_obs = num_features * num_past_days

# Split into training and test set
n_train_days = int(0.9 * len(data))
train = values[:n_train_days, :]
test = values[n_train_days:, :]

train_X, train_y = train[:, :num_obs], train[:, -num_features:]  # Ensure train_y has shape (n_samples, num_features)
test_X, test_y = test[:, :num_obs], test[:, -num_features:]  # Ensure test_y has shape (n_samples, num_features)

train_X = train_X.reshape((train_X.shape[0], num_past_days, num_features))
test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))

## Second: Set up the model and train it
# Setup the model
model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.GRU(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(num_features))  # Output layer with 5 units
model.compile(loss='mae', optimizer='adam')
 
# Train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=5, validation_data=(test_X, test_y), verbose=1, shuffle=False)

# Plot the loss function
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

## NEXT: make predictions and plot one of the features
# Predictions
yhat_test = model.predict(test_X)
yhat_test = yhat_test[:, 4].reshape((len(yhat_test),1))
yhat_train = model.predict(train_X)
yhat_train = yhat_train[:, 4].reshape((len(yhat_train),1))

test_X1 = test_X.reshape((test_X.shape[0], num_past_days * num_features))
train_X1 = train_X.reshape((train_X.shape[0], num_past_days * num_features))

# Invert scaling for forecast
inv_yhat_test = np.concatenate((yhat_test, test_X1[:, -(num_features-1):]), axis=1)
inv_yhat_train = np.concatenate((yhat_train, train_X1[:, -(num_features-1):]), axis=1)
inv_yhat_test = scaler.inverse_transform(inv_yhat_test)
inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
inv_yhat_test = inv_yhat_test[:, 0]
inv_yhat_train = inv_yhat_train[:, 0]

mse, rmse, mae = calculate_accuracy(list(original_data['Adj_Close'][num_past_days:]), list(inv_yhat_train) + list(inv_yhat_test))
mse = round(mse, 2)
rmse = round(rmse, 2)
mae = round(mae, 2)

# Plot result
plt.plot(original_data.index[num_past_days:], original_data['Adj_Close'][num_past_days:], label='Target')
plt.plot(original_data.index[num_past_days:], list(inv_yhat_train) + list(inv_yhat_test), label='Predicted')
plt.legend()
plt.xlabel('Date')
plt.ylabel('BTC/USD')
plt.show()
plt.close()

# FINALLY: we perform a sequential rolling prediction for the next 60 days with 5 features
n_days = 180
last_sequence = test_X[-1].reshape((1, num_past_days, num_features))

rolling_predictions = []

for day in range(n_days):
    # Predict the next day's 5 features
    next_pred = model.predict(last_sequence)[0]
    print(next_pred)
    
    # Store the prediction
    rolling_predictions.append(next_pred)
    
    # Create the new input sequence by appending the prediction
    next_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred.reshape(1, 1, num_features)), axis=1)
    last_sequence = next_sequence

# Convert the list of rolling predictions into a NumPy array
rolling_predictions = np.array(rolling_predictions)

# Invert scaling for the rolling predictions
inv_rolling_predictions = scaler.inverse_transform(rolling_predictions)

# Extract the prediction for the 'Adj_Close' column (assumed to be the 5th feature)
inv_rolling_adj_close = inv_rolling_predictions[:, 4]

# Create a date range for the next 60 days
last_date = original_data.index[-1]
prediction_dates = pd.date_range(last_date, periods=n_days+1)[1:]

# Plot the rolling predictions
plt.plot(original_data.index[num_past_days:], original_data['Adj_Close'][num_past_days:], label='Target')
plt.plot(prediction_dates, inv_rolling_adj_close, label='Rolling Predictions', color='orange')
plt.axvline(x=original_data.index[-1], label='Prediction Start', ymin=0.1, ymax=0.75, linestyle='--')
plt.legend()
plt.xlabel('Date')
plt.ylabel('BTC/USD')
plt.title('Bitcoin Price Prediction with Rolling Forecast (Adj_Close)')
plt.show()

print('MSE = ',mse, 'RMSE = ', rmse, 'MAE = ', mae)