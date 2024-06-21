import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Dataset
data_path = 'Datasets/'
turbidity_training_clean = pd.read_csv(data_path + 'dataset.csv')
turbidity_training_clean.set_index('Date', inplace=True)

# Preprocessing and Input Parameters
class_threshold_1 = float(input("Enter the minimum value for class 1: "))
class_threshold_2 = float(input("Enter the minimum value for class 2: "))

class2_peaks = 0
class1_peaks = 0
class0_peaks = 0

# Input training ranges
col_names = [
    "Turbidity", "Ocean currents speed", "Ocean currents direction", 
    "Wind speed", "Wind direction", "Dust", "Nitrogen dioxide", 
    "Sea temperature", "Sulphur dioxide", "Salinity"
]
max_col = []
min_col = []

for name in col_names:
    max_val = float(input(f"Enter the maximum training value of {name}: "))
    min_val = float(input(f"Enter the minimum training value of {name}: "))
    max_col.append(max_val)
    min_col.append(min_val)

# Preprocess training data
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
    max_value = max_col[i]
    min_value = min_col[i]
    np_for_training_ampliated_scaled[:, i] = (np_for_training_ampliated[:, i] - min_value) / (max_value - min_value)

np_for_training_ampliated_scaled[:, -1] = np_for_training_ampliated[:, -1]

# Model Parameters
num_classes = 3
n_future = int(input("Enter the number of periods in the future you want to predict: "))
n_past = int(input("Enter the number of periods you want to use in each prediction: "))

# Prepare training data for LSTM
trainX = []
trainY = []

for i in range(len(np_for_training_ampliated_scaled) - n_future - n_past + 1):
    trainX.append(np_for_training_ampliated_scaled[i:i + n_past, 0:np_for_training_ampliated_scaled.shape[1]])
    trainY.append(np_for_training_ampliated_scaled[i + n_past + n_future - 1:i + n_past + n_future, -1])

X, Y = np.array(trainX), np.array(trainY)

# Split into training and validation sets
Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=0.05, shuffle=False)

# Build LSTM model
tf.keras.backend.clear_session()
tf.random.set_seed(0)
initializer1 = tf.keras.initializers.GlorotNormal()

model = Sequential([
    LSTM(64, activation='relu', input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True, kernel_initializer=initializer1),
    Dropout(0.05),
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.1),
    LSTM(256, activation='relu', return_sequences=True),
    Dropout(0.15),
    LSTM(256, activation='relu', return_sequences=True),
    Dropout(0.15),
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.1),
    LSTM(64, activation='relu', return_sequences=False),
    Dense(num_classes, activation='softmax')
])

optimizer_1 = tf.keras.optimizers.Adam()

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer_1, metrics=['accuracy'])
model.summary()

# Class weights for imbalance
class_weight = {
    0: (class0_peaks + class1_peaks + class2_peaks) / class0_peaks,
    1: (class0_peaks + class1_peaks + class2_peaks) / class1_peaks * np.exp(-0.05),
    2: (class0_peaks + class1_peaks + class2_peaks) / class2_peaks * np.exp(-0.1)
}

# Train the model
Intended_batch_size = 1000
Epochs = 100

batch_size_calculated = Intended_batch_size
for i in range(0, Intended_batch_size):
    if np.mod(Xtrain.shape[0], (Intended_batch_size + i)) == 0:
        batch_size_calculated = Intended_batch_size + i
        break

steps_per_epoch = int(Xtrain.shape[0] / batch_size_calculated)

history = model.fit(Xtrain, Ytrain, epochs=Epochs, batch_size=batch_size_calculated, validation_data=(Xval, Yval), class_weight=class_weight)

# Save trained model
model_path = 'Models/'
model.save(model_path + 'trained_model_classification.keras')

# Training summary
train_loss = history.history['loss']
validation_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plotting training history
fig, axs = plt.subplots(2, figsize=(10, 8))
fig.suptitle('Training History')
axs[0].plot(train_loss, label='Training Loss')
axs[0].plot(validation_loss, label='Validation Loss')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epoch')
axs[0].legend()

axs[1].plot(train_accuracy, label='Training Accuracy')
axs[1].plot(validation_accuracy, label='Validation Accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].legend()

plt.tight_layout()
plt.show()


