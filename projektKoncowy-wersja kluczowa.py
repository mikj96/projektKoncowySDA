#0. IMPORTY BIBLIOTEK

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import glob
import folium
import plotly.express as px
import plotly.graph_objs as go
import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import *
import webbrowser

#0. ROBOCZE NAZWY MIESIĘCY + ŚCIEŻKA GDZIE POBRALIŚMY DANE

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
filepath = fr"C:\\Users\Enter\OneDrive\Pulpit\smogData\\"

#1. ZNAJDUJEMY WSZYSTKIE NAZWY SENSORÓW
for month in months:
    data = pd.read_csv(fr"{filepath}{month}-2017.csv")
    sensors = []
    for first_row_name in data.columns:
        first_row_name
        sensors.append(first_row_name[0:3])

sensors = pd.unique(sensors)
print('(1/5) All unique sensor names saved.')


#2. ZAMIENIAMY NAZWY SENSORÓW NA INT - UNIKAMY NAZW LITEROWYCH
real_sensors = []
for value in sensors:
    try:
        real_sensors.append(int(value))
    except ValueError:
        pass
print('(2/5) Changing sensor names to integer values completed.')


#3. ZAPISYWANIE MIESIĘCZNYCH ODCZYTÓW DLA KAŻDEGO SENSORA
for month in months:
    data = pd.read_csv(fr"{filepath}{month}-2017.csv")
    for sensor_name in real_sensors:
        df_new = data[
            [f'UTC time', f'{sensor_name}_temperature', f'{sensor_name}_humidity', f'{sensor_name}_pressure',
             f'{sensor_name}_pm1', f'{sensor_name}_pm25',
             f'{sensor_name}_pm10']]
        df_new.to_csv(path_or_buf=fr"{filepath}sensors score\\{sensor_name}{month}.csv")
print('(3/5) Saving data for each month, for each sensor Completed.')

#4. MERGOWANIE MIESIĘCZNYCH TABEL W JEDNĄ ROCZNĄ DLA KAŻDEGO SENSORA
months.pop(0)
for sensor in real_sensors:
    data = pd.read_csv(fr"{filepath}sensors score\\{sensor}january.csv")
    for month in months:
        data_new = pd.read_csv(fr"{filepath}sensors score\\{sensor}{month}.csv")
        data = pd.concat([data, data_new])
    data.to_csv(path_or_buf=fr"{filepath}\\sensors score\\sensor{sensor}.csv")
print('(4/5) Merging month results completed')

#5. USUWANIE DWÓCH NIEPOTRZEBNYCH KOLUMN, REDUKCJA DANYCH PRZEZ UŚREDNIENIE ODCZYTÓW DZIENNYCH
for sensor in real_sensors:
    data = pd.read_csv(fr"{filepath}\\sensors score\\sensor{sensor}.csv")
    del data['Unnamed: 0.1']
    del data['Unnamed: 0']
    for record in data['UTC time']:
        data = data.replace(record, record[:10])
    data = data.groupby('UTC time').mean(numeric_only=True)
    data.to_csv(path_or_buf=fr"{filepath}sensors score\\sensor{sensor}_mean.csv")
print('(5/5) Saving mean values for each sensor completed.')

#6. USUWANIE PLIKÓW POMOCNICZYCH

months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
for sensor in real_sensors:
    for month in months:
        if os.path.exists(fr"{filepath}sensors score\\{sensor}{month}.csv"):
            os.remove(fr"{filepath}sensors score\\{sensor}{month}.csv")
print('(6/7) Deleting data for each month completed')

for sensor in real_sensors:
    if os.path.exists(fr"{filepath}sensors score\\sensor{sensor}.csv"):
        os.remove(fr"{filepath}sensors score\\sensor{sensor}.csv")
print('(7/7) Deleting auxilary data completed.')

