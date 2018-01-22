import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the trainin data
dataset_train = pd.read_csv('train.csv')
train_set = dataset_train.iloc[:,7:8].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)

# creating a training set with 72 timesteps taking the last 3  days data over 24 hours i.e 3*24
X_train = []
y_train = []

for i in range(72,len(train_set)-1):
    X_train.append(training_set_scaled[i-72:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# initializing the RNN and adding layers
regressor = Sequential()
regressor.add(LSTM(units=50,return_sequences= True, input_shape= (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences= True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences= True))
regressor.add(Dropout(0.2))

# adding the fourth and final LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

## adding the output layer
regressor.add(Dense(units=1))

## Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

## fitting the RNN over training set
regressor.fit(X_train,y_train,epochs=3, batch_size = 72)

## loading test data
dataset_test = pd.read_csv('test.csv')

# Predicting the units of electricity
inputs = dataset_train.iloc[:,7:8]
inputs = inputs[len(train_set)-8496:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test=[]
for i in range(72,len(dataset_test)+72):
    X_test.append(inputs[i-72:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_units = regressor.predict(X_test)
predicted_units = sc.inverse_transform(predicted_units)


