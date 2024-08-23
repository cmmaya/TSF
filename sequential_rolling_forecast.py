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
file_name = 'Eth_Daily_from_2023_07.csv'
file_path = os.path.join(folder_name, file_name)

close = 'Adj_Close'
Output_address = "C:\\Users\\camil\\Documents\\Juanp\\TSF\\results\\"
cols = ["Open", "High", "Low", "Close", "Volume"]
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
data = data[cols].dropna()

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
num_past_days = 12
data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
values = data.values
num_obs = num_features * num_past_days

# Split into training and test set
n_train_days = int(0.8 * len(data))
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
# model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(num_features))  # Output layer with 5 units
model.compile(loss='mae', optimizer='adam')
 
# Train the model
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), verbose=1, shuffle=False)

# Plot the loss function
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

# FINALLY: we perform a sequential rolling prediction for the next 60 days with 5 features
# Perform prediction for the next n_days and calculate accuracy
n_days = 1
predictions = []
actuals = []

for i in range(0, len(test_X) - n_days, n_days):
    last_sequence = test_X[i].reshape((1, num_past_days, num_features))
    temp_predictions = []
    
    for day in range(n_days):
        next_pred = model.predict(last_sequence)[0]
        temp_predictions.append(next_pred)
        next_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred.reshape(1, 1, num_features)), axis=1)
        last_sequence = next_sequence
    
    predictions.append(np.array(temp_predictions))
    actuals.append(test_y[i:i+n_days])

# Convert the list of predictions and actuals to NumPy arrays
predictions = np.array(predictions).reshape(-1, num_features)
actuals = np.array(actuals).reshape(-1, num_features)

# Invert scaling for predictions and actuals
inv_predictions = scaler.inverse_transform(predictions)
inv_actuals = scaler.inverse_transform(actuals)

# Calculate accuracy for the 'Adj_Close' column (assumed to be the 5th feature)
inv_predictions_adj_close = inv_predictions[:, 3]
inv_actuals_adj_close = inv_actuals[:, 3]

mse, rmse, mae = calculate_accuracy(inv_actuals_adj_close, inv_predictions_adj_close)
mse = round(mse, 2)
rmse = round(rmse, 2)
mae = round(mae, 2)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plot the predictions vs actual values
plt.figure(figsize=(14, 7))

plt.plot(original_data.index[-len(inv_actuals_adj_close):], inv_actuals_adj_close, label='Actual Close', color='blue')
plt.plot(original_data.index[-len(inv_actuals_adj_close):], inv_predictions_adj_close, label='Predicted Close', color='red')
plt.title('Actual vs Predicted Adj_Close')
plt.xlabel('Time')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()