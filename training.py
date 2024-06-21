import pandas as pd
import numpy as np
from numpy import savetxt
import math
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import History
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras import initializers
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import datetime
from datetime import datetime, timezone, timedelta
from matplotlib.dates import date2num

# Load DATASET

data_path = 'Datasets/'

# Read from CSV file
turbidity_training_clean = pd.read_csv(data_path + 'dataset.csv')
turbidity_training_clean.set_index('Date', inplace=True)

## PREPROCESSING DATA MODEL Cross Entropy (one prediction in n_future periods in future)

# Parameters to input:
class_threshold_1 = input("Enter the minimum value for class 1: ")
class_threshold_2 = input("Enter the minimum value for class 2: ")

class2_peaks = 0
class1_peaks = 0
class0_peaks = 0
num_classes = 3

# INTRODUCTION OF THE MAXIMES AND MINIMS USED IN TRAINING
max_col_1 = input("Enter the maximum training value of Turbidity: ")
min_col_1 = input("Enter the minimum training value of Turbidity: ")
max_col_2 = input("Enter the maximum training value of Ocean currents speed: ")
min_col_2 = input("Enter the minimum training value of Ocean currents speed: ")
max_col_3 = input("Enter the maximum training value of Ocean currents direction: ")
min_col_3 = input("Enter the minimum training value of Ocean currents direction: ")
max_col_4 = input("Enter the maximum training value of Wind speed: ")
min_col_4 = input("Enter the minimum training value of Wind speed: ")
max_col_5 = input("Enter the maximum training value of Wind direction: ")
min_col_5 = input("Enter the minimum training value of Wind direction: ")
max_col_6 = input("Enter the maximum training value of Dust: ")
min_col_6 = input("Enter the minimum training value of Dust: ")
max_col_7 = input("Enter the maximum training value of Nitrogen dioxide: ")
min_col_7 = input("Enter the minimum training value of Nitrogen dioxide: ")
max_col_8 = input("Enter the maximum training value of Sea temperature: ")
min_col_8 = input("Enter the minimum training value of Sea temperature: ")
max_col_9 = input("Enter the maximum training value of Sulphur dioxide: ")
min_col_9 = input("Enter the minimum training value of Sulphur dioxide: ")
max_col_10 = input("Enter the maximum training value of Salinity: ")
min_col_10 = input("Enter the minimum training value of Salinity: ")

np_for_training = np.array(turbidity_training_clean)
np_for_training_ampliated = np.c_[np_for_training, np.ones(len(np_for_training))]

for i in range(len(np_for_training_ampliated)):
    if np_for_training_ampliated[i, 0] >= class_threshold_2:
        np_for_training_ampliated[i, -1] = 2
        class2_peaks += 1
    elif np_for_training_ampliated[i, 0] >= class_threshold_1:
        np_for_training_ampliated[i, -1] = 1
        class1_peaks += 1
    else:
        np_for_training_ampliated[i, -1] = 0
        class0_peaks += 1

np_for_training_ampliated_scaled = np.zeros_like(np_for_training_ampliated)

for i in range(10):
    col_name = f"col_{i+1}" 
    max_value = globals()['max_' + col_name]
    min_value = globals()['min_' + col_name]
    np_for_training_ampliated_scaled[:, i] = (np_for_training_ampliated[:, i] - min_value) / (max_value - min_value)

np_for_training_ampliated_scaled[:, -1] = np_for_training_ampliated[:, -1]

trainX = []
trainY = []

n_future = input("Enter the number of periods in the future you want to predict: ")
n_past = input("Enter the number of periods you want to use in each prediction: ")

for i in range(len(np_for_training_ampliated_scaled) - n_future - n_past + 1):
    trainX.append(np_for_training_ampliated_scaled[i:i + n_past, 0:np_for_training_ampliated_scaled.shape[1]])
    trainY.append(np_for_training_ampliated_scaled[i + n_past + n_future - 1:i + n_past + n_future, -1])
X, Y = np.array(trainX), np.array(trainY)


Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.05, shuffle=False)  # before model building

# Build the model

tf.keras.backend.clear_session()
tf.random.set_seed(0)
initializer1 = tf.keras.initializers.GlorotNormal()

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True, kernel_initializer=initializer1))
model.add(Dropout(0.05))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256, activation='relu', return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(256, activation='relu', return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))
optimizer_1 = keras.optimizers.Adam()

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=optimizer_1, metrics=['accuracy'])
model.summary()

class_weight = {
    0: (class0_peaks + class1_peaks + class2_peaks) / class0_peaks,
    1: (class0_peaks + class1_peaks + class2_peaks) / class1_peaks * np.exp(-0.05),
    2: (class0_peaks + class1_peaks + class2_peaks) / class2_peaks * np.exp(-0.1)
}

# Training the model

Intended_batch_size = 1000
Epochs = 100

# Calculate the nearest and appropriate batch_size to that which we have requested
batch_size_calculated = Intended_batch_size
for i in range(0, Intended_batch_size):
    if np.mod(Xtrain.shape[0], (Intended_batch_size + i)) == 0:
        batch_size_calculated = Intended_batch_size + i
        break
steps_per_epoch = int(Xtrain.shape[0] / batch_size_calculated)

history = model.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=batch_size_calculated, validation_data=(Xval, Yval), class_weight=class_weight)

model.save('trained_model.keras')

# Training summary

train_loss = history.history['loss']
validation_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

fig_chl = plt.figure(figsize=(10, 5), constrained_layout=True)
spec = fig_chl.add_gridspec(ncols=10, nrows=10)

a0 = fig_chl.add_subplot(spec[:, :5])
a0.plot(train_loss, label='Training')
a0.plot(validation_loss, label='Validation')
a0.set_ylabel('Error', fontsize=12)
a0.set_xlabel('Epoch', fontsize=12)
a0.legend(fontsize=12)

a1 = fig_chl.add_subplot(spec[:, 5:])
a1.plot(train_accuracy, label='Training')
a1.plot(validation_accuracy, label='Validation')
a1.set_ylabel('Accuracy', fontsize=12)
a1.set_xlabel('Epoch', fontsize=12)
a1.legend(fontsize=12)

plt.show()

