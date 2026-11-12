# Lekcja 12 - Zadanie 12
# Uruchom: python zadanie_12.py
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
    df = add_regions(housing_data())
    g = df.groupby('Region')['MedHouseVal']
    q1 = g.transform(lambda x: x.quantile(.25))
    q3 = g.transform(lambda x: x.quantile(.75))
    iqr = q3 - q1
    df['Is_Outlier'] = ((df['MedHouseVal'] < q1 - 1.5 * iqr) |
                        (df['MedHouseVal'] > q3 + 1.5 * iqr)).astype(int)
    print('Outliery według regionu:')
    print(df.groupby('Region')['Is_Outlier'].sum())
    stats = g.agg(min='min', Q1=lambda x: x.quantile(.25), med='median',
                  Q3=lambda x: x.quantile(.75), max='max')
    print('\nStatystyki 5-punktowe:')
    print(stats.round(3))
    lo, hi = df['MedHouseVal'].min(), df['MedHouseVal'].max()
    def pos(v):
        return int((v - lo) / (hi - lo) * 49) if hi > lo else 0
    for region, row in stats.iterrows():
        line = [' '] * 50
        for j in range(pos(row['min']), pos(row['max']) + 1): line[j] = '-'
        for j in range(pos(row['Q1']), pos(row['Q3']) + 1): line[j] = '='
        line[pos(row['med'])] = '*'
        line[pos(row['min'])] = '|'
        line[pos(row['max'])] = '|'
        print(f'{region} |{"".join(line)}|')
