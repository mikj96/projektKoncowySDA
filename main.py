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


def zmiennaFunkcja(month):

    data = pd.read_csv(f"C:\\Users\Enter\OneDrive\Pulpit\smogData\\{month}-2017.csv")
    data = data.dropna(thresh = 1, axis='columns')
    sensors = []
    for column_name in data.columns:
        sensors.append(column_name[0:3])

    #bo jest wartosc UTC na pierwszej pozycji
    sensors.pop(0)

    #bo wartosci sie powtarzaja
    sensors = list(pd.unique(sensors))
    amount = len(sensors)

    counter = len(sensors) + 1
    correct = 0
    faulty = 0
    number = 1

    while counter >= 2:
        for name in sensors:
            try:
                df_new = data[
                    [f'{name}_temperature', f'{name}_humidity', f'{name}_pressure', f'{name}_pm1', f'{name}_pm25',
                     f'{name}_pm10']]
                df_new.to_csv(path_or_buf=f"C:\\Users\Enter\OneDrive\Pulpit\smogData\sensors score\\{name}{month}.csv")
                print(f'({number}/{amount})Zakończono procedurę dla czujnika nr {name} ')
                sensors.pop(sensors.index(f'{name}'))
                counter = counter - 1
                correct += 1
                number += 1

            except KeyError:
                print(f'({number}/{amount})Niekompletny fragment tabeli dla czujnika nr {name}')
                sensors.pop(sensors.index(f'{name}'))
                counter = counter - 1
                faulty += 1
                number += 1
            pass
    else:
        print(f'Zakończono procedure dla {amount} sensorów.',
              f'Pobrano odczyty z {correct}/{amount} sensorów. {faulty}/{amount} zawiera niekompletne lub niepoprawne dane.',
              sep='\n')



'''
    #nie działa bez usuwania obecnej lokalizacji
    filepath = str(f'{month}')
    parent_dir = 'C:\\Users\Enter\OneDrive\Pulpit\smogData'
    path = os.path.join(parent_dir, filepath)
    os.mkdir(path)
    print(f"(1/1) Directory {parent_dir}\{filepath} has been created.")
'''
def inicjalizacjaFunkcji():
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    counter = 1
    for month in months:
        zmiennaFunkcja(month)
        print(f'({counter}/12) Zakończono procedurę zapisu danych pogodowych dla miesiąca: {month}')
        counter += 1
    return print('(1/1) Zakończono procedurę zapisu danych pogodowych dla wszystkich miesięcy.')

inicjalizacjaFunkcji()