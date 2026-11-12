# Lekcja 12 - Zadanie 17
# Uruchom: python zadanie_17.py
# Wymagane: pip install pandas numpy
# Zadania z Titanic wymagaja dostepu do internetu.

import pandas as pd
import numpy as np

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def titanic_data():
    return pd.read_csv(TITANIC_URL)

def housing_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(32.5, 41.5, n)
    longitude = rng.uniform(-123.5, -114.5, n)
    med_inc = rng.gamma(shape=2.0, scale=1.8, size=n) + 0.5
    house_age = rng.integers(1, 55, n)
    ave_rooms = rng.normal(5.5, 1.5, n).clip(1, 15)
    ave_bedrms = (ave_rooms * rng.uniform(0.15, 0.25, n)).clip(0.5, 5)
    population = rng.integers(50, 5000, n)
    ave_occup = rng.normal(3.0, 0.8, n).clip(1, 10)
    med_house_val = (med_inc * 0.6 + (latitude - 36) * 0.15 - house_age / 100 + rng.normal(0, 0.3, n)).clip(0.15, 5.0)
    return pd.DataFrame({'MedInc': med_inc.round(4), 'HouseAge': house_age,
        'AveRooms': ave_rooms.round(4), 'AveBedrms': ave_bedrms.round(4),
        'Population': population, 'AveOccup': ave_occup.round(4),
        'Latitude': latitude.round(4), 'Longitude': longitude.round(4),
        'MedHouseVal': med_house_val.round(4)})

def add_regions(df):
    df = df.copy()
    north = df['Latitude'] >= 36
    east = df['Longitude'] >= -119
    df['Region'] = np.select([north & east, north & ~east, ~north & east, ~north & ~east],
                              ['NE', 'NW', 'SE', 'SW'], default='Unknown')
    return df

def bars(series, scale=2):
    for name, value in series.items():
        if pd.notna(value):
            print(f'{str(name):22s} | {"█" * int(max(0, value) // scale)} {value:.2f}%')


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    continents = {'Europa': {'Polska': ['Warszawa', 'Kraków', 'Gdańsk'],
                              'Niemcy': ['Berlin', 'Hamburg', 'Monachium'],
                              'Francja': ['Paryż', 'Lyon', 'Marsylia']},
                  'Azja': {'Japonia': ['Tokio', 'Osaka', 'Kioto'],
                           'Chiny': ['Pekin', 'Szanghaj', 'Shenzhen'],
                           'Indie': ['Delhi', 'Mumbaj', 'Bengaluru']},
                  'Ameryka Płn.': {'USA': ['Nowy Jork', 'Chicago', 'Boston'],
                                   'Kanada': ['Toronto', 'Ottawa', 'Vancouver'],
                                   'Meksyk': ['Meksyk', 'Guadalajara', 'Monterrey']}}
    rows = []
    for continent, countries in continents.items():
        for country, cities in countries.items():
            for city in cities:
                for value in rng.integers(100, 2000, size=50):
                    rows.append((continent, country, city, int(value)))
    df = pd.DataFrame(rows, columns=['Kontynent', 'Kraj', 'Miasto', 'Sprzedaż'])
    a = df.groupby('Kontynent')['Sprzedaż'].sum()
    b = df.groupby(['Kontynent', 'Kraj'])['Sprzedaż'].sum()
    c = df.groupby(['Kontynent', 'Kraj', 'Miasto'])['Sprzedaż'].sum()
    print('Kontynenty:\n', a.to_frame('Sprzedaż'))
    print('Kraje:\n', b.to_frame('Sprzedaż'))
    print('Miasta:\n', c.to_frame('Sprzedaż'))
    total = a.sum()
    print('\nDRZEWO SPRZEDAŻY:')
    for continent, value in a.items():
        print(f'{continent}: {value:,} ({value / total:.2%} całości)')
        for (con, country), country_val in b.items():
            if con != continent: continue
            print(f'  {country}: {country_val:,} ({country_val / value:.2%} kontynentu)')
            for (con2, country2, city), city_val in c.items():
                if con2 == continent and country2 == country:
                    print(f'    {city}: {city_val:,} ({city_val / country_val:.2%} kraju)')
