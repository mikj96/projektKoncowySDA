import pandas as pd

data = pd.read_csv(f"C:\\Users\Enter\OneDrive\Pulpit\smogData\\april-2017.csv")
data = data.dropna(thresh = 1, axis='columns')
sensors = []
for column_name in data.columns:
    sensors.append(column_name[0:3])
sensors.pop(0)
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
              'november', 'december']
months.pop(0)
sensors = pd.unique(sensors)

for sensor in sensors:
    try:
        dataF = pd.read_csv(f"C:\\Users\Enter\OneDrive\Pulpit\smogData\sensors score\\{sensor}january.csv")
        for month in months:
            try:
                dane = pd.read_csv(f'C:\\Users\Enter\OneDrive\Pulpit\smogData\sensors score\\{sensor}{month}.csv')
                tabela = pd.concat([dataF, dane])
                print(tabela)

            except FileNotFoundError:
                print(f'Brak odczytów z sensora {sensor} dla miesiaca {month}')
                pass
        print(dataF)
    except FileNotFoundError:
        print(f'Brak odczytów z sensora {sensor} dla miesiaca january')
        pass


