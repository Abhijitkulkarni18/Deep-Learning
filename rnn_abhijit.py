# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 18:43:55 2018

@author: abhij
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('C:\\python files\\machine learning\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\Recurrent_Neural_Networks\\TATASTEEL.csv')
training_set = dataset_train.iloc[:, 1:2].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = training_set_scaled[0:4511]
y_train = training_set_scaled[1:4512]

X_train = np.reshape(X_train, (4511, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

#test_set = pd.read_csv('Google_Stock_Price_Test.csv')
dataset_train = pd.read_csv('C:\\python files\\machine learning\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\Recurrent_Neural_Networks\\TATASTEEL.csv')
real_stock_price = dataset_train.iloc[:, 1:2].values
#prediction
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(4507,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

'''
forecast = predicted_stock_price

forecast_pred = []
for i in range(100):
    
    inputs_pred = forecast
    inputs_pred = sc.transform(inputs_pred)
    
    inputs_pred = inputs_pred[-1:,:]
    
    inputs_pred = np.reshape(inputs_pred,(1,1,1))

    predicted = regressor.predict(inputs_pred)
    predicted = sc.inverse_transform(predicted)
    forecast = np.append(forecast ,predicted, axis = 0)
    forecast_pred.append(predicted)
'''
'''
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values

predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

plt.plot(real_stock_price_train, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

'''









