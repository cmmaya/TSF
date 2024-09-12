# TSF
Time Series Forecast ML and statistical algorithms for Bitcoin and Ethereum.

## Simulation file
Here we run a simulation of an investing strategy based on the following rules:

* The function loops through each date starting from the 15th
* If you are currently holding the stock (curr_holding is True), the function sells the stock at the current price and updates your cash balance based on the return from the trade.
* If you are not holding the stock, the function checks whether the predicted return exceeds the threshold thresh.
* If it does, you buy the stock at the current price. If the prediction exceeds the thresh, you buy the stock.
* When buying, the function records the date and price.
* When selling, the function calculates the return from the trade and updates the total amount of money.
* All buy and sell events are recorded in events_list.
* The function plots the stock prices over time.
* Vertical lines are drawn on the plot to indicate buy and sell events.
* The area between a buy and a sell event is shaded green if the trade was profitable and red if it was not.
* The function prints out the total amount of money left at the end of the simulation.
* The function returns the final amount.

The model is either an LSTM using the past 20 days, 20 epochs and 5 batch size or ARIMA of order (5,1,0)

### Example:
Here is a result example using LSTM:
![image](https://github.com/user-attachments/assets/ee264ed4-0278-431f-a94c-c73a64800607)
