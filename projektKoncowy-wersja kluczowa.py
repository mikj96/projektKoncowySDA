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

7# RYSOWANIE WYKRESU

for sensor in real_sensors:
    dane = pd.read_csv(fr"C:\\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor{sensor}_mean.csv", parse_dates=True,
                       index_col='UTC time')
    dane = dane.dropna(how='any')
    dane.rename(columns={f'{sensor}_temperature': 'Temperatura', f'{sensor}_humidity': 'Wilgotnosc'
        , f'{sensor}_pressure': 'Cisnienie', f'{sensor}_pm1': 'Stezenie PM1', f'{sensor}_pm25': 'Stezenie PM25',
                         f'{sensor}_pm10': 'Stezenie PM10'}, inplace=True)

    fig, ax = plt.subplots()
    # zmienna_wykresu = str(input('temperature/humidity/pressure'))
    ax.plot(dane.index, dane['Temperatura'], color='r')
    ax.xaxis.set_tick_params(rotation=90)

    # rysujemy wykresy dla każdego stężenia PMI
    # color - kolor markera, linestyle - styl linii, label - nazwa krzywej; potrzebna do legendy
    ax2 = ax.twinx()
    ax2.plot(dane.index, dane['Stezenie PM1'], color='b', linestyle='--', label='PMI 1')
    ax2.plot(dane.index, dane['Stezenie PM25'], color='c', linestyle='--', label='PMI 2.5')
    ax2.plot(dane.index, dane['Stezenie PM10'], color='g', linestyle='--', label='PMI 10')
    ax2.legend()
    ax2.set_ylabel('Stężenia PMI')
    ax.set_title(f'Wykres temperatura/stężenie pyłków dla sensora:{sensor}')
    ax.set_xlabel('Data')
    ax.set_ylabel('Temperatura')

    ax.grid()
    #plt.show() #na razie zablokowane żeby nie rysowało wykresów

#8. PREDYKCJA
#8.1. REGRESJA LINIOWA
print('(1/2) Inititating procedure to calculate linear regression.')
pmi_category = str(input('(2/2) Choose concentration metric for linear regression: Stezenie PM1, Stezenie PM25 ,Stezenie PM10'))

def linear_regr(pmi_cat):
    r2_scorelist = {}
    counter = 0
    tasks = len(real_sensors)
    for sensor in real_sensors:
        try:
            dane = pd.read_csv(fr"C:\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor{sensor}_mean.csv",
                               parse_dates=True,
                               index_col='UTC time')
            dane = dane.dropna(how='any')
            dane.rename(columns={f'{sensor}_temperature': 'Temperatura', f'{sensor}_humidity': 'Wilgotnosc'
                , f'{sensor}_pressure': 'Cisnienie', f'{sensor}_pm1': 'Stezenie PM1', f'{sensor}_pm25': 'Stezenie PM25',
                                 f'{sensor}_pm10': 'Stezenie PM10'}, inplace=True)

            dane = dane.astype(int)
            X = dane[['Temperatura', 'Wilgotnosc', 'Cisnienie']]
            y = dane[f'{pmi_cat}']

            model = LinearRegression().fit(X, y)
            y_predict = model.predict(X)
            r2 = r2_score(y, y_predict)
            r2_scorelist.update({sensor: r2})
            counter += 1
            print(f'({counter}/{tasks}) Adding score for {sensor} to database.')
        except ValueError:
            counter += 1
            print(f'{counter}/{tasks})Sensor {sensor} has incomplete or faulty data.')
            continue
    highest_score = np.max(list(r2_scorelist.values()))
    lowest_score = np.min(list(r2_scorelist.values()))
    mean_score = np.mean(list(r2_scorelist.values()))
    if mean_score <= 0.5 and mean_score > 0:
        print(f'(1/5) Highest prediction score was: {highest_score}.')
        print(f'(2/5) Lowest prediction score was: {lowest_score}.')
        print(f'(3/5) Overall average R2 score was: {mean_score} which is not satysfing.',
              "(4/5) It's not enough for appropriate model.", sep='\n')
    else:
        print(f'(1/5)  Highest prediction score was: {highest_score}.')
        print(f'(2/5)  Lowest prediction score was: {lowest_score}.')
        print(f'(3/5)  Overall average R2 score was: {mean_score} which is satysfing.',
              "(4/5) It's enough for appropriate model.", sep='\n')
    return print('(5/5) Procedure finished.')

linear_regr(pmi_category)

#8.2. REGRESJA WIELOMIANOWA
print('(1/2) Inititating procedure to calculate polynomial regression.')
pmi_category = str(input('(2/2) Choose concentration metric for polynomial regression: Stezenie PM1, Stezenie PM25 ,Stezenie PM10'))

def poly_regr(pmi_cat):
    r2_scorelist = {}
    r2_highest = {}
    counter = 0
    tasks = len(real_sensors)
    for sensor in real_sensors:
        try:
            dane = pd.read_csv(fr"C:\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor{sensor}_mean.csv",
                               parse_dates=True,
                               index_col='UTC time')
            dane = dane.dropna(how='any')
            dane.rename(columns={f'{sensor}_temperature': 'Temperatura', f'{sensor}_humidity': 'Wilgotnosc'
                , f'{sensor}_pressure': 'Cisnienie', f'{sensor}_pm1': 'Stezenie PM1', f'{sensor}_pm25': 'Stezenie PM25',
                                 f'{sensor}_pm10': 'Stezenie PM10'}, inplace=True)

            dane = dane.astype(int)
            X = dane[['Temperatura', 'Wilgotnosc', 'Cisnienie']]
            y = dane[f'{pmi_cat}']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            poly = PolynomialFeatures(degree=3)
            X_poly = poly.fit_transform(X.values)
            model = LinearRegression().fit(X_poly, y)
            predict_data = X_test.values
            predict_data_poly = poly.transform(predict_data)
            y_predict = model.predict(predict_data_poly)
            r2 = r2_score(y_test, y_predict)
            r2_scorelist.update({sensor: r2})
            if r2 >= 0.7:
                r2_highest.update({sensor: r2})
            counter += 1
            print(f'({counter}/{tasks}) Adding score for {sensor} to database.')
        except ValueError:
            counter += 1
            print(f'{counter}/{tasks})Sensor {sensor} has incomplete or faulty data.')
            continue
    highest_score = np.max(list(r2_scorelist.values()))
    lowest_score = np.min(list(r2_scorelist.values()))
    mean_score = np.mean(list(r2_scorelist.values()))
    if mean_score <= 0.5 and mean_score > 0:
        if bool(r2_highest) == False:
            print(f'(1/5) Highest prediction score was: {highest_score}.')
            print(f'(2/5) Lowest prediction score was: {lowest_score}.')
            print(f'(3/5) Overall average R2 score was: {mean_score} which is not satysfing.',
                  "(4/5) It's not enough for appropriate model.", sep='\n')
        else:
            print(f'(0/5) Group of sensors which gave appropriate results: {r2_highest}')
            print(f'(1/5) Highest prediction score was: {highest_score}.')
            print(f'(2/5) Lowest prediction score was: {lowest_score}.')
            print(f'(3/5) Overall average R2 score was: {mean_score} which is not satysfing.',
                  "(4/5) It's not enough for appropriate model.", sep='\n')
    else:
        print(f'(1/5)  Highest prediction score was: {highest_score}.')
        print(f'(2/5)  Lowest prediction score was: {lowest_score}.')
        print(f'(3/5)  Overall average R2 score was: {mean_score} which is satysfing.',
              "(4/5) It's enough for appropriate model.", sep='\n')
    return print('(5/5) Procedure finished.')

poly_regr(pmi_category)