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


#0. FILEPATH - LOKALIZACJA POBRANYCH
#TRZEBA UTWORZYĆ FOLDER SENSORS SCORE WEWNĄTRZ FOLDERU SMOG DATA

filepath = fr"C:\\Users\Enter\OneDrive\Pulpit\smogData\\"
def creating_folder(filepath):
    directory = 'sensors score'
    path = os.path.join(filepath, directory)
    return print('(1/1) Created folder for data.')

#1. FUNKCJA ITERUJĄCA
#BĘDZIE ITEROWAĆ DANE PO MIESIĄCU ZAWARTYM W NAZWIE PLIKU
#CEL: FUNKCJA POMOCNICZA - GWARANTUJE NAM, ŻE WSZYSTKIE MIESIĄCE ZOSTANĄ WZIĘTE POD UWAGĘ W NASTĘPNEJ FUNKCJI; ZAPEWNIA
#ZACHOWANIE CHRONOLOGII ZAPISU
def month_iteration():
    #miesiace zebrane tabelarycznie w formacie pasującym do danych z tabeli
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    counter = 1
    for month in months:
        #uruchomienie funkcji dla każdego miesiąca z osobna
        grouping_sensors(month)
        print(f'({counter}/12) Saving weather data for month: {month}')
        counter += 1
    return print('(1/1) Finished saving weather data for all months.')
#do czego służy tak naprawdę? - do odpalenia funkcji grouping_sensors c


#2. ZAPISYWANIE DANYCH Z WIELU MIESIĘCY OSOBNO DLA KAŻDEGO SENSORA
#CEL: WYSELEKCJONOWANIE TABEL DLA KAŻDEGO MIESIĄCA I SENSORA, W CELU POŁĄCZENIA ICH PÓŹNIEJ W POJEDYNCZE TABELE, OSOBNE
#DLA KAŻDEGO SENSORA

def grouping_sensors(month):

    data = pd.read_csv(fr"{filepath}{month}-2017.csv")
    sensors = []
    for first_row_name in data.columns:
        sensors.append(first_row_name[0:3])

    #bo jest wartosc UTC na pierwszej pozycji
    sensors.pop(0)

    #bo wartosci sie powtarzaja
    sensors = list(pd.unique(sensors))
    amount = len(sensors)

    #zmienne pomocnicze do opisywania etapu działania programu
    counter = len(sensors) + 1
    correct = 0
    faulty = 0
    number = 1

    while counter >= 2:
        for sensor_name in sensors:
            try:
                df_new = data[
                    [f'UTC time',f'{sensor_name}_temperature', f'{sensor_name}_humidity', f'{sensor_name}_pressure', f'{sensor_name}_pm1', f'{sensor_name}_pm25',
                     f'{sensor_name}_pm10']]
                df_new.to_csv(path_or_buf=fr"{filepath}sensors score\\{sensor_name}{month}.csv")
                print(f'({number}/{amount})Finished procedure for sensor number: {sensor_name}.')

                sensors.pop(sensors.index(f'{sensor_name}'))
                counter = counter - 1
                correct += 1
                number += 1

            except KeyError:
                print(f'({number}/{amount})Incomplete table for sensor number: {sensor_name}.')
                sensors.pop(sensors.index(f'{sensor_name}'))
                counter = counter - 1
                faulty += 1
                number += 1
            pass
    else:
        print(f'Finished procedure for {amount} sensors.',
              f'Received results for {correct}/{amount} sensors. ({faulty}/{amount}) are incomplete or faulty.',
              sep='\n')



#3. ZAPISYWANIE TABEL W NOWYM FORMACIE
#ZBIORCZE ZAPISANIE DANYCH TABELARYCZNYCH DLA POSZCZEGÓLNYCH MIESIECY
#CEL: ZAPISANIE DANYCH TABELARYCZNYCH DLA POJEDYNCZEGO SENSORA
def merging_sensors():
    #odczyt danych tabelarycznych w celu ustalenia nazw sensorów
    print('(1/3) Starting the bulk file saving procedure.')
    data = pd.read_csv(fr"{filepath}april-2017.csv")
    data = data.dropna(thresh=1, axis='columns')
    sensors = []
    for column_name in data.columns:
        sensors.append(column_name[0:3])
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    print('(2/3) Finalizing the preparatory processes.')
    sensor_list = []
    for sensor in sensors:
        auxiliary_list = []
        for month in months:
            try:
                dane = pd.read_csv(fr"{filepath}sensors score\\{sensor}{month}.csv")
                auxiliary_list.append(f'{sensor}{month}')
                print(f'')
            except FileNotFoundError:
                pass
        sensor_list.append(auxiliary_list)

    #sensor_list - lista list w której zlokalizowane są wszystkie "sprawne" sensory
    for list in sensor_list:
        try:
            dane = pd.read_csv(fr"{filepath}sensors score\\{list[0]}.csv")
            dane['Unnamed: 0'] = f'{(list[0])[3:]}'
            print(f'(0/{len(list)}) Saving readings for the sensor {(list[0])[:3]}')
            counter = 1
            while counter <= len(list):
                print(f'({counter}/{len(list)}) Adding results from month {(list[counter])[3:]} to the table of sensor {(list[0])[:3]}.')
                daneF = pd.read_csv(fr"{filepath}sensors score\\{list[counter]}.csv")
                daneF['Unnamed: 0'] = f'{(list[counter])[3:]}'
                dane = pd.concat([dane, daneF])
                counter += 1
        except IndexError:
            pass
        finally:
            try:
                for record in dane['UTC time']:
                    dane = dane.replace(record, record[:10])

                dane = dane.groupby('UTC time').mean(numeric_only= True)
                dane.to_csv(
                    path_or_buf=fr"{filepath}sensors score\\sensor{list[0][:3]}.csv")
                print(f'({counter}/{len(list)}) Saving results for sensor {(list[0])[:3]} finished.')
            except IndexError:
                pass
    return print('(3/3) Bulk formulation of tables completed.')

#4. USUWANIE PLIKÓW POMOCNICZYCH



#5. ZAINICJOWANIE PROGRAMU

month_iteration()
merging_sensors()


#6. BŁĘDY W PROGRAMIE:
#FILEPATH - DODANIE ZMIENNEJ KTÓRA SPRAWI ŻE PO WPROWADZENIU LOKALIZACJI PLIKÓW MOŻEMY WYKORZYTAĆ PROGRAM NA WIELU
#URZĄDZENIACH
#FUNKCJE - ZAGNIEŻDŻENIE FUNKCJI W INNYCH FUNKCJACH W CELU OSZCZĘDNOŚCI MIEJSCA





'''7# RYSOWANIE WYKRESU
dane = pd.read_csv(fr"C:\\Users\Enter\OneDrive\Pulpit\smogData\sensors score\sensor226.csv", parse_dates= True)
#usuwanie/modyfikowanie kolumn
del dane['Unnamed: 0.1']
del dane['Unnamed: 0']
#dane = dane.rename(columns={"Unnamed: 0": "month"})
dane = dane.dropna(how ='any')

#ZOSTAWIAMY TYLKO PIERWSZE 10 ZNAKÓW DATY, ŻEBY POZBYĆ SIĘ WSKAZAŃ GODZINOWYCH
#DZIĘKI TEMU ZA POMOCĄ KOMENDY GROUPBY MOŻEMY UŚREDNIĆ WARTOŚCI PARAMETRÓW
#DLA JEDNEGO DNIA
for record in dane['UTC time']:
    dane = dane.replace(record,record[:10])

dane = dane.groupby('UTC time').mean()
print(dane)

fig,ax = plt.subplots()


#zmienna_wykresu = str(input('temperature/humidity/pressure'))
ax.plot(dane.index,dane[f'226_temperature'], color = 'r')
ax.xaxis.set_tick_params(rotation = 90)

#rysujemy wykresy dla każdego stężenia PMI
#color - kolor markera, linestyle - styl linii, label - nazwa krzywej; potrzebna do legendy
ax2 = ax.twinx()
ax2.plot(dane.index,dane['226_pm1'], color = 'b', linestyle = '--', label = 'PMI 1')
ax2.plot(dane.index,dane['226_pm25'], color = 'c', linestyle = '--', label = 'PMI 2.5')
ax2.plot(dane.index,dane['226_pm10'], color = 'g', linestyle = '--', label = 'PMI 10')
ax2.legend()
ax2.set_ylabel('Stężenia PMI')
ax.set_title('Pierwszy sensor - 226')
ax.set_xlabel('Data')
#ax.set_ylabel(f'{zmienna_wykresu}')

#plt.show()
ax.grid()
plt.show()



#8.1. REGRESJA LINIOWA
X = dane[['226_temperature']]
y = dane['226_pm1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model_linear = LinearRegression()
model_linear.fit(np.array(X_train).reshape(-1,1), y_train)
predict_data = np.array([[0],[5]])
print(model_linear.predict(predict_data))

#8.2 REGRESJA WIELOMIANOWA




poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

predict_data = np.array([[10],[5]]).reshape(-1,1)
print(predict_data)
predict_data_poly = poly.transform(predict_data)

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2)

print(X.shape, y.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)





predict_data = np.array([[10],[5]])

predict_data_poly = poly.transform(predict_data)

model.predict(predict_data_poly)
'''