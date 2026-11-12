# Lekcja 12 - Zadanie 20
# Uruchom: python zadanie_20.py
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
    import time
    np.random.seed(42)
    N = 1_000_000
    raw = pd.DataFrame({'Region': np.random.choice(['North', 'South', 'East', 'West'], N),
                        'Produkt': np.random.choice(list('ABCDE'), N),
                        'Sprzedaż': np.random.randint(10, 1000, N),
                        'Koszt': np.random.randint(5, 500, N)})
    optimized = raw.copy()
    for col in ['Region', 'Produkt']:
        optimized[col] = optimized[col].astype('category')
    for col in ['Sprzedaż', 'Koszt']:
        optimized[col] = optimized[col].astype('int16')
    def mb(df):
        return df.memory_usage(deep=True).sum() / 1024**2
    def aggregate(df):
        return df.groupby('Region', observed=True).agg(
            Sprzedaz_suma=('Sprzedaż', 'sum'), Sprzedaz_srednia=('Sprzedaż', 'mean'),
            Koszt_suma=('Koszt', 'sum'), Liczba=('Sprzedaż', 'size'))
    def benchmark(df):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = aggregate(df)
            times.append(time.perf_counter() - start)
        return sum(times) / len(times), result
    before, after = mb(raw), mb(optimized)
    t_before, agg_raw = benchmark(raw)
    t_after, agg_opt = benchmark(optimized)
    assert np.allclose(agg_raw.sort_index(), agg_opt.sort_index())
    print(f'Pamięć przed: {before:.2f} MB')
    print(f'Pamięć po zmianie typów: {after:.2f} MB')
    print(f'Oszczędność: {(1-after/before)*100:.2f}%')
    print(f'Czas agregacji przed (3 próby): {t_before:.4f} s')
    print(f'Czas agregacji po (3 próby): {t_after:.4f} s')
    print(f'Zmiana czasu: {(1-t_after/t_before)*100:.2f}% (dodatnia = szybciej)')
    print(f'Pamięć samej agregacji: {mb(agg_opt):.5f} MB')
    print('\nAgregacja według regionu:')
    print(agg_opt.round(2))
    assert (1-after/before) >= .5
