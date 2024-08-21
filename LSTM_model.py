import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error
from math import sqrt

import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Define the folder and file name
folder_name = 'datasets'
file_name = 'Dataset_01.csv'
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

# Here we are Transforming the data for the Neural Network in a lag based matrix (nth:matrix).
def apply_transform(data, n: int):
    middle_data = []
    target_data = []
    for i in range(n, len(data)):
        input_sequence = data[i-n:i]  
        middle_data.append(input_sequence) 
        target_data.append(data[i])
    middle_data = np.array(middle_data).reshape((len(middle_data), n, 1))
    target_data = np.array(target_data)
    return middle_data,target_data

# This the LSTM model training Function 
def LSTM(train,n : int, number_nodes, learning_rate, epochs, batch_size):
   middle_data, target_data = apply_transform(train, n)
   model = tf.keras.Sequential([
      tf.keras.layers.Input((n,1)),
      tf.keras.layers.LSTM(number_nodes,input_shape=(n, 1)),
      tf.keras.layers.Dense(units = number_nodes,activation = "relu"),
      tf.keras.layers.Dense(units = number_nodes,activation = "relu"),
      tf.keras.layers.Dense(1)
   ])
   model.compile(loss = 'mse',optimizer = tf.keras.optimizers.Adam(learning_rate),metrics = ["mean_absolute_error"])

   print(f"middle_data shape: {middle_data.shape}")
   print(f"target_data shape: {target_data.shape}")
   print(f"LSTM input shape: {model.input_shape}")
   history = model.fit(middle_data,target_data,epochs = epochs,batch_size = batch_size,verbose = 1, shuffle=True)
   full_predictions = model.predict(middle_data).flatten()
   # Save the model
   model.save('models/LSTM_one_layer.h5')
   return model,history,full_predictions

# ARIMA Model Function for Predicting the possible ERRORS from LSTM Model.
def ARIMA_Model(train,len_test,ord):
   model = ARIMA(train, order = ord)
   model = model.fit()
   predictions = model.predict(start = len(train),end = len(train) + len_test ,type='levels')
   full_predictions = model.predict(start = 0,end = len(train)-1,type='levels')
   return model,predictions,full_predictions

# ARIMA Parameter Selection and PACF & ACF
def Parameter_calculation(data):
   finding = auto_arima(data,trace = True)
   ord = finding.order
   return ord

# The Final Prediction : LSTM predicted value + ARIMA predicted Error value
def Final_Predictions(predictions_errors,predictions, days):
   final_values = []
   for i in range(days):
      final_values.append(predictions_errors[i] + predictions[i])
   return final_values

def Error_Evaluation(train_data,predict_train_data,n:int):
   errors = []
   for i in range(len(predict_train_data)):
      err = train_data[n + i] - predict_train_data[i]
      errors.append(err)
   return errors

def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse,rmse,mae

def plot_train_test(train, test):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Train Set')
    plt.plot(test.index, test, label='Test Set', color='orange')
    plt.title('Train and Test Data')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    address = Output_address + 'plot_train_test' + ".jpg"
    plt.savefig(address)

def plot_predictions(train, predictions,title):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Actual')
    plt.plot(train.index, predictions, label='Predicted', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close-Price')
    address = Output_address + title + ".jpg"
    address = Output_address + 'plot_predictions' + ".jpg"
    plt.savefig(address)

def plot_performance(history):
   plt.plot(history.history['loss'])
   plt.title('Model Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   address = Output_address + 'plot_performance' + ".jpg"
   plt.savefig(address)

def plot_final_predictions(test, final_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, final_predictions, label='Corrected Prediction', color='green')
    plt.title('Final Predictions with Error Correction')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    address = Output_address + 'Final Predictions with Error Correction' + ".jpg"
    plt.savefig(address)

def plot_arima_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color=['blue', 'orange', 'green'])
    plt.title('ARIMA Model Accuracy Metrics')
    address = Output_address + 'Model Accuracy Metrics' + ".jpg"
    plt.savefig(address)

def main():
   days = 12
   n = 12

   # Separate train and test datasets
   data = data_loader() 
   train_len_val = len(data) - days
   train,test = data[close].iloc[0:train_len_val],data[close].iloc[train_len_val:]

   # Single Batch Training and Prediction
   model, history, full_predictions = LSTM(train, days, 32, 0.001, 15, 32)
   plot_predictions(train[days:], full_predictions,"LSTM PREDICTIONS VS ACTUAL Values For TRAIN Data Set")
   plot_performance(history)

   # Sequential Rolling Forecasting
   last_sequence = train[-n:].values.reshape((1, n, 1))  
   predictions = []
   for i in range(days+1):
      next_prediction = model.predict(last_sequence).flatten()[0]
      predictions.append(next_prediction)
      if i < len(test):
         actual_value = test.iloc[i]
         new_row = np.append(last_sequence[:, 1:, :], np.array([[[actual_value]]]), axis=1)
      else:
         new_row = np.append(last_sequence[:, 1:, :], np.array([[[next_prediction]]]), axis=1)        
      last_sequence = new_row.reshape((1, n, 1))
   plot_predictions(test,predictions[:-1], "LSTM Predictions VS Actual Values")

   for i in range(days):
      actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
      print(f"Day {i+1} => ACTUAL VALUE : {actual_value} | PREDICTED VALUE : {predictions[i]}\n")

   # Arima model
   errors_data = Error_Evaluation(train,full_predictions,n)
   ord = Parameter_calculation(errors_data)
   Arima_Model,predictions_errors,full_predictions_errors = ARIMA_Model(errors_data,len(test),ord)
   for i in range(len(predictions_errors)):
      print(f"{i+1} : {predictions_errors[i]}\n")

   final_predictions = Final_Predictions(predictions_errors,predictions,days)
   plot_final_predictions(test[:days], final_predictions[:days])

   print(Arima_Model.summary())
   arima_mse, arima_rmse, arima_mae = calculate_accuracy(errors_data, full_predictions_errors)
   plot_arima_accuracy(arima_mse, arima_rmse, arima_mae)
   

main()