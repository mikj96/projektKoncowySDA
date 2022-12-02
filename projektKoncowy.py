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


#0. FILEPATH - LOKALIZACJA POBRANYCH

filepath = fr"C:\\Users\Enter\OneDrive\Pulpit\smogData\\"


#1. FUNKCJA ITERUJĄCA
#BĘDZIE ITEROWAĆ DANE PO MIESIĄCU ZAWARTYM W NAZWIE PLIKU
#CEL: FUNKCJA POMOCNICZA - GWARANTUJE NAM, ŻE WSZYSTKIE MIESIĄCE ZOSTANĄ WZIĘTE POD UWAGĘ W NASTĘPNEJ FUNKCJI; ZAPEWNIA
#ZACHOWANIE CHRONOLOGII ZAPISU
def inicjalizacjaFunkcji():
    #miesiace zebrane tabelarycznie w formacie pasującym do danych z tabeli
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    counter = 1
    for month in months:
        #uruchomienie funkcji dla każdego miesiąca z osobna
        zmiennaFunkcja(month)
        print(f'({counter}/12) Zakończono procedurę zapisu danych pogodowych dla miesiąca: {month}')
        counter += 1
    return print('(1/1) Zakończono procedurę zapisu danych pogodowych dla wszystkich miesięcy.')



#2. ZAPISYWANIE DANYCH Z WIELU MIESIĘCY OSOBNO DLA KAŻDEGO SENSORA
#CEL: WYSELEKCJONOWANIE TABEL DLA KAŻDEGO MIESIĄCA I SENSORA, W CELU POŁĄCZENIA ICH PÓŹNIEJ W POJEDYNCZE TABELE, OSOBNE
#DLA KAŻDEGO SENSORA

def zmiennaFunkcja(month):

    data = pd.read_csv(fr"{filepath}{month}-2017.csv")
    data = data.dropna(thresh = 1, axis='columns')
    sensors = []
    for column_name in data.columns:
        sensors.append(column_name[0:3])

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
        for name in sensors:
            try:
                df_new = data[
                    [f'{name}_temperature', f'{name}_humidity', f'{name}_pressure', f'{name}_pm1', f'{name}_pm25',
                     f'{name}_pm10']]
                df_new.to_csv(path_or_buf=fr"{filepath}sensors score\\{name}{month}.csv")
                print(f'({number}/{amount})Zakończono procedurę dla czujnika nr {name}.')

                sensors.pop(sensors.index(f'{name}'))
                counter = counter - 1
                correct += 1
                number += 1

            except KeyError:
                print(f'({number}/{amount})Niekompletny fragment tabeli dla czujnika nr {name}.')
                sensors.pop(sensors.index(f'{name}'))
                counter = counter - 1
                faulty += 1
                number += 1
            pass
    else:
        print(f'Zakończono procedure dla {amount} sensorów.',
              f'Pobrano odczyty z {correct}/{amount} sensorów. ({faulty}/{amount}) zawiera niekompletne lub niepoprawne dane.',
              sep='\n')



#3. ZAPISYWANIE TABEL W NOWYM FORMACIE
#ZBIORCZE ZAPISANIE DANYCH TABELARYCZNYCH DLA POSZCZEGÓLNYCH MIESIECY
#CEL: ZAPISANIE DANYCH TABELARYCZNYCH DLA POJEDYNCZEGO SENSORA
def modyfikowaniePlikow():
    #odczyt danych tabelarycznych w celu ustalenia nazw sensorów
    print('(1/3) Rozpoczynanie procedury zbiorczego zapisu plików.')
    data = pd.read_csv(fr"{filepath}april-2017.csv")
    data = data.dropna(thresh=1, axis='columns')
    sensors = []
    for column_name in data.columns:
        sensors.append(column_name[0:3])
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
    print('(2/3) Finalizowanie procesów przygotowawczych.')
    sensor_list = []
    for sensor in sensors:
        lista_pomocnicza = []
        for month in months:
            try:
                dane = pd.read_csv(fr"{filepath}sensors score\\{sensor}{month}.csv")
                lista_pomocnicza.append(f'{sensor}{month}')
                print(f'')
            except FileNotFoundError:
                pass
        sensor_list.append(lista_pomocnicza)

    #sensor_list - lista list w której zlokalizowane są wszystkie "sprawne" sensory
    for list in sensor_list:
        try:
            dane = pd.read_csv(fr"{filepath}sensors score\\{list[0]}.csv")
            dane['Unnamed: 0'] = f'{(list[0])[3:]}'
            print(f'(0/{len(list)}) Zapisywanie odczytów dla sensora {(list[0])[:3]}')
            counter = 1
            while counter <= len(list):
                print(f'({counter}/{len(list)}) Dodawanie miesiąca {(list[counter])[3:]} do tabeli sensora {(list[0])[:3]}.')
                daneF = pd.read_csv(fr"{filepath}sensors score\\{list[counter]}.csv")
                daneF['Unnamed: 0'] = f'{(list[counter])[3:]}'
                dane = pd.concat([dane, daneF])
                counter += 1
        except IndexError:
            pass
        finally:
            try:
                dane.to_csv(
                    path_or_buf=fr"{filepath}sensors score\\sensor{list[0][:3]}.csv")
                print(f'({counter}/{len(list)}) Zapisywanie odczytów dla sensora {(list[0])[:3]} zakończone.')
            except IndexError:
                pass
    return print('(3/3) Zbiorcze formułowanie tabel zakończone.')

#4. USUWANIE PLIKÓW POMOCNICZYCH



#5. ZAINICJOWANIE PROGRAMU
inicjalizacjaFunkcji()
modyfikowaniePlikow()


#6. BŁĘDY W PROGRAMIE:
#FILEPATH - DODANIE ZMIENNEJ KTÓRA SPRAWI ŻE PO WPROWADZENIU LOKALIZACJI PLIKÓW MOŻEMY WYKORZYTAĆ PROGRAM NA WIELU
#URZĄDZENIACH
#FUNKCJE - ZAGNIEŻDŻENIE FUNKCJI W INNYCH FUNKCJACH W CELU OSZCZĘDNOŚCI MIEJSCA