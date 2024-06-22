import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load DATASET
data_path = 'Datasets/'
turbidity_training_clean = pd.read_csv(data_path + 'dataset.csv')
turbidity_training_clean.set_index('Date', inplace=True)

# INTRODUCTION OF THE MAXIMES AND MINIMS USED IN TRAINING (You may also automate this process)
max_col = []
min_col = []
col_names = [
    'Turbidity', 'Ocean currents speed', 'Ocean currents direction',
    'Wind speed', 'Wind direction', 'Dust', 'Nitrogen dioxide',
    'Sea temperature', 'Sulphur dioxide', 'Salinity'
]

for name in col_names:
    max_val = float(input(f"Enter the maximum training value of {name}: "))
    min_val = float(input(f"Enter the minimum training value of {name}: "))
    max_col.append(max_val)
    min_col.append(min_val)

np_for_training = np.array(turbidity_training_clean)
np_for_training_ampliated_scaled = np.zeros_like(np_for_training)

for i in range(10):
    np_for_training_ampliated_scaled[:, i] = (np_for_training[:, i] - min_col[i]) / (max_col[i] - min_col[i])

trainX = []
trainY = []

# Model Parameters
n_future = int(input("Enter the number of periods in the future you want to predict: "))
n_past = int(input("Enter the number of periods you want to use in each prediction: "))

for i in range(len(np_for_training_ampliated_scaled) - n_future - n_past + 1):
    trainX.append(np_for_training_ampliated_scaled[i:i + n_past, 0:np_for_training_ampliated_scaled.shape[1]])
    trainY.append(np_for_training_ampliated_scaled[i + n_past + n_future - 1:i + n_past + n_future, 0])

X, Y = np.array(trainX), np.array(trainY)
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.05, shuffle=False)

# Build LSTM model
tf.keras.backend.clear_session()
tf.random.set_seed(0)
initializer1 = tf.keras.initializers.GlorotNormal() 

# Get the number of layers and layer sizes from the user
num_layers = int(input("Enter the number of LSTM layers: "))
if num_layers <= 0:
    raise ValueError("The number of layers must be a positive integer.")
layer_sizes = input(f"Enter a vector of {num_layers} sizes for the LSTM layers, separated by commas (e.g. 64,128,256,...): ")
layer_sizes = [int(size) for size in layer_sizes.split(",")]
if len(layer_sizes) != num_layers:
    raise ValueError(f"You must enter exactly {num_layers} sizes.")

# Create the model using the user-specified layer sizes
model = Sequential()

# Add LSTM and Dropout layers
for i, size in enumerate(layer_sizes):
    return_sequences = True if i < num_layers - 1 else False
    if i == 0:
        model.add(LSTM(size, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=return_sequences, kernel_initializer=initializer1))
    else:
        model.add(LSTM(size, activation='relu', return_sequences=return_sequences))
    model.add(Dropout(0.05 + 0.05 * i))  # Increase dropout with each layer, adjust as necessary

# Add the final Dense layer
model.add(Dense(Ytrain.shape[1], activation='relu'))

optimizer_1 = tf.keras.optimizers.Adam()

model.compile(loss='mean_squared_error', optimizer=optimizer_1, metrics=['accuracy'])
model.summary()

# Training the model
batch_size_calculated = 1000  # Set your intended batch size
history = model.fit(
    Xtrain, Ytrain,
    epochs=100,
    batch_size=batch_size_calculated,
    validation_data=(Xval, Yval),
    verbose=1
)

# Save trained model
model_path = 'Models/'
model.save(model_path + 'trained_model_regression.keras')

# Training summary visualization
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
