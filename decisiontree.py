
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
from sklearn.tree import DecisionTreeRegressor
from sklearn import *
import webbrowser
from sklearn.metrics import *
from sklearn.model_selection import KFold

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

sensor = 170

dane = pd.read_csv(fr"C:\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor{sensor}_mean.csv",
    parse_dates=True, index_col='UTC time')
dane = dane.dropna(how='any')
dane.rename(columns={f'{sensor}_temperature': 'Temperatura', f'{sensor}_humidity': 'Wilgotnosc'
                , f'{sensor}_pressure': 'Cisnienie', f'{sensor}_pm1': 'Stezenie PM1', f'{sensor}_pm25': 'Stezenie PM25',
                                 f'{sensor}_pm10': 'Stezenie PM10'}, inplace=True)

dane = dane.astype(int)
X = dane[['Temperatura', 'Wilgotnosc', 'Cisnienie']]
y = dane[f'Stezenie PM10']

print('(3/3) Drzewko decyzyjne.')
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
model_tree = DecisionTreeRegressor()
model_tree.fit(X_train.values, y_train.values)
predict_data = X_test.values
y_predict = model_tree.predict(predict_data)
r2 = r2_score(y_test, y_predict)
print(r2)

