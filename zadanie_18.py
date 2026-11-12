# Lekcja 12 - Zadanie 18
# Uruchom: python zadanie_18.py
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
    df = titanic_data().dropna(subset=['Age']).copy()
    manual = [[0, 12, 18, 35, 60, 100], [0, 8, 15, 25, 40, 60, 100],
              [0, 6, 14, 22, 30, 45, 65, 100]]
    def evaluate(label, bins):
        result = df.groupby(bins, observed=True)['Survived'].agg(['count', 'mean'])
        spread = result['mean'].max() - result['mean'].min()
        print(f'\n{label} (rozpiętość = {spread:.3f}):')
        print(result.round(3))
        return spread
    scores = {'equal-width': evaluate('equal-width', pd.cut(df['Age'], bins=5)),
              'equal-frequency': evaluate('equal-frequency', pd.qcut(df['Age'], q=5, duplicates='drop'))}
    for i, edges in enumerate(manual, 1):
        scores[f'manual-{i}'] = evaluate(f'manual-{i}', pd.cut(df['Age'], edges, include_lowest=True))
    print('\nNajlepsza strategia według rozpiętości:', max(scores, key=scores.get))
    print('Uwaga: ocena na tych samych danych jest eksploracyjna, nie jest walidacją predykcji.')
