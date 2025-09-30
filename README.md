## Devloped by: SUDHARSAN S
## Register Number: 212224040334
## Date: 

# Ex.No: 6                   HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
```

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("Amazon Sale Report.csv", parse_dates=["Date"])

if "Amount" not in data.columns:
    data["Amount"] = np.random.randint(100, 500, size=len(data))

data.set_index("Date", inplace=True)
data_weekly = data["Amount"].resample("W").sum()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_weekly.values.reshape(-1, 1)).flatten(),
    index=data_weekly.index
)

plt.figure(figsize=(10,5))
scaled_data.plot(color="blue")
plt.title("Scaled Weekly Sales Data (Date vs Mean Amount)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Mean Amount", fontsize=12)
plt.grid(True)
plt.show()

if len(data_weekly) >= 8:
    decomposition = seasonal_decompose(data_weekly, model="additive", period=4)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.suptitle("Seasonal Decomposition (Weekly Additive Model)", fontsize=16)
    for ax, label in zip(fig.axes, ["Observed", "Trend", "Seasonal", "Residual"]):
        ax.set_ylabel(f"{label} (Mean)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
    plt.show()

scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

if len(train_data) >= 4:
    model_add = ExponentialSmoothing(
        train_data, trend='add', seasonal='add', seasonal_periods=4
    ).fit()

    test_predictions_add = model_add.forecast(steps=len(test_data))

    plt.figure(figsize=(10,5))
    train_data.plot(label="Train Data", color="blue")
    test_predictions_add.plot(label="Test Predictions", color="green")
    test_data.plot(label="Test Data", color="red")
    plt.legend()
    plt.title("Train vs Test Predictions (Weekly Sales)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Mean Amount", fontsize=12)
    plt.grid(True)
    plt.show()

    rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

if len(data_weekly) >= 4:
    final_model = ExponentialSmoothing(
        data_weekly, trend='add', seasonal='mul', seasonal_periods=4
    ).fit()

    final_predictions = final_model.forecast(steps=8)

    plt.figure(figsize=(10,5))
    data_weekly.plot(label="Original Weekly Sales", color="blue")
    final_predictions.plot(label="Final Predictions", color="orange")
    plt.legend()
    plt.title("Final Forecast (Weekly Sales - Date vs Mean Amount)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Mean Amount", fontsize=12)
    plt.grid(True)
    plt.show()

```

### OUTPUT:
## Scaled Weekly Sales Data (Date vs Mean Amount)

![alt text](<Screenshot 2025-09-30 133452.png>)

## Seasonal Decomposition (Weekly Additive Model)

![alt text](<Screenshot 2025-09-30 133504.png>)

## Train vs Test Predictions (Weekly Sales)

![alt text](<Screenshot 2025-09-30 133527.png>)

## RMSE Evaluation

![alt text](<Screenshot 2025-09-30 133547.png>)

## Final Forecast (Weekly Sales - Date vs Mean Amount)

 ![alt text](<Screenshot 2025-09-30 133600.png>)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
