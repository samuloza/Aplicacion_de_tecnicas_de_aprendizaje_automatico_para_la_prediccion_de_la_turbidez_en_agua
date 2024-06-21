import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load validation dataset
data_path = 'Datasets/'
turbidity_validation_clean = pd.read_csv(data_path + 'validation_dataset.csv', delimiter=';', decimal=',')
turbidity_validation_clean.set_index('Date', inplace=True)

# Parameters for classification thresholds and scaling
class_threshold_1 = float(input("Enter the minimum value for class 1: "))
class_threshold_2 = float(input("Enter the minimum value for class 2: "))

# Counters for peaks in each class
class2_peaks = 0
class1_peaks = 0
class0_peaks = 0

# Introduction of max and min values used in training
max_col = []
min_col = []
col_names = [
    "Turbidity", "Ocean currents speed", "Ocean currents direction", 
    "Wind speed", "Wind direction", "Dust", "Nitrogen dioxide", 
    "Sea temperature", "Sulphur dioxide", "Salinity"
]

for name in col_names:
    max_val = float(input(f"Enter the maximum training value of {name}: "))
    min_val = float(input(f"Enter the minimum training value of {name}: "))
    max_col.append(max_val)
    min_col.append(min_val)

# Preprocess data similar to training
np_for_validation = np.array(turbidity_validation_clean)
np_for_validation_ampliated = np.c_[np_for_validation, np.ones(len(np_for_validation))]

for i in range(len(np_for_validation_ampliated)):
    if np_for_validation_ampliated[i, 0] >= class_threshold_2:
        np_for_validation_ampliated[i, -1] = 2
        class2_peaks += 1
    elif np_for_validation_ampliated[i, 0] >= class_threshold_1:
        np_for_validation_ampliated[i, -1] = 1
        class1_peaks += 1
    else:
        np_for_validation_ampliated[i, -1] = 0
        class0_peaks += 1

np_for_validation_ampliated_scaled = np.zeros_like(np_for_validation_ampliated)

for i in range(10):
    np_for_validation_ampliated_scaled[:, i] = (np_for_validation_ampliated[:, i] - min_col[i]) / (max_col[i] - min_col[i])

np_for_validation_ampliated_scaled[:, -1] = np_for_validation_ampliated[:, -1]

# Input parameters for prediction
n_future = int(input("Enter the number of periods in the future you want to predict: "))
n_past = int(input("Enter the number of periods you want to use in each prediction: "))

# Prepare X_predict and Y_predict for prediction
X_predict = []
Y_predict = []

for i in range(len(np_for_validation_ampliated_scaled) - n_future - n_past + 1):
    X_predict.append(np_for_validation_ampliated_scaled[i:i + n_past, 0:np_for_validation_ampliated_scaled.shape[1]])
    Y_predict.append(np_for_validation_ampliated_scaled[i + n_past + n_future - 1:i + n_past + n_future, -1])

X_predict, Y_predict = np.array(X_predict), np.array(Y_predict)

# Load pre-trained model
model_path = 'Models/'
model = tf.keras.models.load_model(model_path + 'trained_model_classification.keras')
model.summary()

# Predictions
predictions = model.predict(X_predict)

# Save predictions to a CSV file
output_path = 'Predictions/'
df_pred = pd.DataFrame(predictions, columns=["Class_0_predicted", "Class_1_predicted", "Class_2_predicted"])
df_pred.to_csv(output_path + 'predicted_outputs_classification.csv', index=False)

# Postprocessing - scaling predictions to classes
predictions_scaled = np.argmax(predictions, axis=1)

# Results
def count_peaks(pred, real, j):
    return np.sum((real == j) & (pred == 0)), np.sum((real == j) & (pred == 1)), np.sum((real == j) & (pred == 2))

for j in range(3):
    count_0, count_1, count_2 = count_peaks(predictions_scaled, Y_predict, j)
    print(f'Number of class {j} peaks predicted as class 0: {count_0}')
    print(f'Number of class {j} peaks predicted as class 1: {count_1}')
    print(f'Number of class {j} peaks predicted as class 2: {count_2}')

