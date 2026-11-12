# Lekcja 12 - Zadanie 15
# Uruchom: python zadanie_15.py
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
    df = housing_data()
    df['lat_bin'] = pd.cut(df['Latitude'], bins=10, labels=False, include_lowest=True)
    df['lon_bin'] = pd.cut(df['Longitude'], bins=10, labels=False, include_lowest=True)
    result = df.groupby(['lat_bin', 'lon_bin']).agg(
        Srednia_cena=('MedHouseVal', 'mean'), Mediana_dochodu=('MedInc', 'median'),
        Populacja=('Population', 'sum'), Liczba=('Population', 'size'))
    print('TOP 10 najdroższych:')
    print(result.nlargest(10, 'Srednia_cena').round(3))
    print('\nTOP 10 najtańszych:')
    print(result.nsmallest(10, 'Srednia_cena').round(3))
    # np.histogram2d daje liczbę obserwacji w komórkach; ceny agregujemy osobno.
    counts, _, _ = np.histogram2d(df['Latitude'], df['Longitude'], bins=10)
    price_grid = result['Srednia_cena'].unstack().reindex(index=range(10), columns=range(10))
    quartiles = price_grid.stack().quantile([.25, .5, .75]).to_numpy()
    print('\nMapa cen (północ u góry): · najniższe, ▪ niskie, ■ wysokie, █ najwyższe')
    for lat in range(9, -1, -1):
        symbols = []
        for lon in range(10):
            val = price_grid.loc[lat, lon]
            symbols.append(' ' if counts[lat, lon] == 0 else
                           '·' if val <= quartiles[0] else
                           '▪' if val <= quartiles[1] else
                           '■' if val <= quartiles[2] else '█')
        print(' '.join(symbols))
