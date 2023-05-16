import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import xgboost as xgb

import CoreFunctions as cf
import os
from binance.client import Client
# load the data
# create a sample dataframe
data = {'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'open': [30000, 31000, 32000],
        'high': [32000, 33000, 34000],
        'low': [29000, 29500, 31000],
        'close': [31000, 32000, 33000],
        'volume': [1000, 2000, 3000]}
df = pd.DataFrame(data)

# save the dataframe to a csv file
df.to_csv('data/Binance_BTCUSDT_d.csv', index=False)

# load the csv file
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df
df = pd.read_csv('data/Binance_BTCUSDT_d.csv')

# print the loaded dataframe
print(df)
candles = cf.load_data('data/Binance_BTCUSDT_d.csv')

# print the first few rows
print(candles.head())
# Your code to read in the candles data

# Check the contents of the candles list
print(candles)

# Create the DataFrame with the correct column names
candles_df = pd.DataFrame(candles, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'extra_column1', 'extra_column2', 'extra_column3', 'extra_column4', 'extra_column5', 'extra_column6'])

# Only select the columns you need
candles_df = candles_df[['date', 'open', 'high', 'low', 'close', 'volume']]

# Call the FeatureCreation function
x = cf.FeatureCreation(candles_df)
# Load API keys from environment variables
api_key = os.getenv('')
api_secret = os.getenv('')

# Create Binance client object
client = Client(api_key, api_secret)
# Load the data and create features/targets
candles = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "12 Dec, 2017", "15 May, 2023")

x = cf.FeatureCreation(candles)
y = cf.CreateTargets(candles,1)
y = y[94:]
x = x[94:len(candles)-1]

# Create spiking neural network model
# Set up the spiking neural network model architecture
num_inputs = x.shape[1]
num_hidden = 100
num_outputs = 1
spike_rates = np.zeros((x.shape[0], num_inputs))

model_spiking = Sequential()
model_spiking.add(Dense(num_hidden, activation='relu', input_dim=num_inputs))
model_spiking.add(Dense(num_outputs, activation='sigmoid'))

# Compile the spiking neural network model
optimizer_spiking = Adam(learning_rate=0.001)
model_spiking.compile(optimizer=optimizer_spiking, loss='binary_crossentropy', metrics=['accuracy'])

# Train the spiking neural network model
es_spiking = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
history_spiking = model_spiking.fit(spike_rates, y, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es_spiking])

# Create Keras model
model_keras = Sequential()
model_keras.add(Dense(32, activation='relu', input_dim=x.shape[1]))
model_keras.add(Dropout(0.5))
model_keras.add(Dense(16, activation='relu'))
model_keras.add(Dropout(0.5))
model_keras.add(Dense(1, activation='sigmoid'))

optimizer_keras = Adam(learning_rate=0.001)
model_keras.compile(optimizer=optimizer_keras, loss='binary_crossentropy', metrics=['accuracy'])

# Train Keras model
es_keras = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
history_keras = model_keras.fit(x, y, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es_keras])

# Create XGBoost model
model_xgb = xgb.XGBClassifier()
model_xgb.fit(x, y)

# Combine Keras, XGBoost, and spiking neural network models
preds_keras = model_keras.predict(x)
preds_xgb = model_xgb.predict(x)
spike_rates = np.zeros((x.shape[0], num_inputs))
preds_spiking = model_spiking.predict(spike_rates)
preds = (preds_keras + preds_xgb + preds_spiking) / 3

# Save the model
model_keras.save("model_keras.h5")
model_xgb.save_model("model_xgb.bin")
model_spiking.save("model_spiking.h5")

# Evaluate the models
mse_keras = model.evaluate(x, y)[0]
mse_xgb = np.mean((preds_xgb - y)**2)
mse_spiking = None  # TODO: Replace with spiking neural model MSE

accuracy_keras = model.evaluate(x, y)[1]
accuracy_xgb = np.mean(preds_xgb == y)
accuracy_spiking = None  # TODO: Replace with spiking neural model accuracy

weight_keras = model.count_params()
weight_xgb = None  # TODO: Replace with XGBoost model weight
weight_spiking = None  # TODO: Replace with spiking neural model weight

quality = 1 - (mse_keras + mse_xgb + mse_spiking) / 3  # Calculate the quality of the model

# Print the results
print("MSE (Keras):", mse_keras)
print("MSE (XGBoost):", mse_xgb)
print("MSE (Spiking):", mse_spiking)
print("Accuracy (Keras):", accuracy_keras)
print("Accuracy (XGBoost):", accuracy_xgb)
print("Accuracy (Spiking):", accuracy_spiking)
print("Weight (Keras):", weight_keras)
print("Weight (XGBoost):", weight_xgb)
print("Weight (Spiking):", weight_spiking)
print("Quality:", quality)
