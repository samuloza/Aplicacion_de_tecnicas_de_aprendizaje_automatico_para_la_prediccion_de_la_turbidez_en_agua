import pandas as pd
import numpy as np
from numpy import savetxt
import math
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy.interpolate import RegularGridInterpolator
import datetime
from datetime import datetime, timezone, timedelta
from matplotlib.dates import date2num

# Extract turbidity data
turbidity_path = 'RAW_Data/'
df_turbidity_raw = pd.read_csv(turbidity_path + 'raw_data.csv', usecols=['Date', 'Turbidity'])

# Preprocess turbidity data
df_turbidity_raw['Date'] = pd.to_datetime(df_turbidity_raw['Date'])
df_turbidity_raw.set_index('Date', inplace=True)
df_turbidity_raw['Turbidity'] = df_turbidity_raw['Turbidity'].apply(pd.to_numeric, errors='coerce')

# Extract and join Meteomatics data
meteo_path = 'RAW_Data/Meteomatics/'
df_meteo_raw = pd.read_csv(meteo_path + 'Meto_Extract.csv', usecols=[
    'validdate', 'ocean_current_speed:kmh', 'ocean_current_direction:d', 'wind_dir_10m:d', 'wind_speed_10m:kmh',
    'dust_0p9um_20um:ugm3', 'no2:ugm3', 't_sea_sfc:C', 'so2:ugm3', 'salinity:psu'
])

# Preprocess Meteomatics data
df_meteo_raw.rename(columns={
    "validdate": "Date",
    "ocean_current_speed:kmh": "Ocean_current_speed(km/h)",
    "ocean_current_direction:d": "Ocean_current_direction(degrees)",
    "wind_dir_10m:d": "Wind_direction(degrees)",
    "wind_speed_10m:kmh": "Wind_speed(km/h)",
    "dust_0p9um_20um:ugm3": "Dust(ug/m^3)",
    "no2:ugm3": "NO2(ug/m^3)",
    "t_sea_sfc:C": "Sea_temperature(C)",
    "so2:ugm3": "SO2(ug/m^3)",
    "salinity:psu": "Salinity(PSU)"
}, inplace=True)
df_meteo_raw['Date'] = pd.to_datetime(df_meteo_raw['Date'])
df_meteo_raw.set_index('Date', inplace=True)

# Data adaptation to five-minute intervals
df_turbidity_raw_mean = df_turbidity_raw.resample('5T').mean()
df_turbidity_raw_mean['Turbidity'].fillna(method='ffill', inplace=True)
turbidity_training = pd.concat([df_turbidity_raw_mean, df_meteo_raw], axis=1)

# Save the dataset
save_path = 'Datasets/'
turbidity_training.to_csv(save_path + 'dataset.csv', index=True, header=True)
