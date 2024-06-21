import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load validation dataset
data_path = 'Datasets/'
turbidity_validation_clean = pd.read_csv(data_path + 'validation_dataset.csv', delimiter=';', decimal=',')
turbidity_validation_clean.set_index('Date', inplace=True)

# INTRODUCTION OF THE MAXIMUMS AND MINIMUMS USED IN TRAINING (You may also automate this process)
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

# Prepare validation data similar to training
np_for_validation = np.array(turbidity_validation_clean)
np_for_validation_ampliated = np.c_[np_for_validation, np.ones(len(np_for_validation))]

for i in range(len(np_for_validation_ampliated)):
    np_for_validation_ampliated[i, :-1] = (np_for_validation_ampliated[i, :-1] - min_col) / (max_col - min_col)

# Input parameters for prediction
n_future = int(input("Enter the number of periods in the future you want to predict: "))
n_past = int(input("Enter the number of periods you want to use in each prediction: "))

# Prepare X_predict for prediction
X_predict = []
for i in range(len(np_for_validation_ampliated) - n_future - n_past + 1):
    X_predict.append(np_for_validation_ampliated[i:i + n_past, :-1])

X_predict = np.array(X_predict)

# Load pre-trained model
model_path = 'Models/'
model = tf.keras.models.load_model(model_path + 'trained_model_regression.keras')
model.summary()

# Predictions
predictions = model.predict(X_predict)

# Save predictions to a CSV file
output_path = 'Predictions/'
df_pred = pd.DataFrame(predictions, columns=["Predicted_turbidity"])
df_pred.to_csv(output_path + 'predicted_outputs_regression.csv', index=False)

print(df_pred)
