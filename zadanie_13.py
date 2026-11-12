# Lekcja 12 - Zadanie 13
# Uruchom: python zadanie_13.py
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
    df = titanic_data()
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False).fillna('Unknown')
    counts = df['Title'].value_counts()
    df['Title'] = df['Title'].where(df['Title'].map(counts) >= 10, 'Rare')
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
    df['Age_Group'] = pd.cut(df['Age'], [0, 12, 18, 35, 60, np.inf], labels=False, include_lowest=True)
    df['Sex_Encoded'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Title_Encoded'] = pd.factorize(df['Title'])[0]
    df['Embarked_Encoded'] = pd.factorize(df['Embarked'])[0]
    df['Avg_Fare_Pclass'] = df.groupby('Pclass')['Fare'].transform('mean')
    df['Avg_Age_Pclass'] = df.groupby('Pclass')['Age'].transform('mean')
    df['Avg_Survival_Title'] = df.groupby('Title')['Survived'].transform('mean')
    df['Age_Std_Pclass'] = df.groupby('Pclass')['Age'].transform(lambda x: (x-x.mean()) / x.std())
    df['Fare_Std_Pclass'] = df.groupby('Pclass')['Fare'].transform(lambda x: (x-x.mean()) / x.std())
    df['Fare_Log'] = np.log1p(df['Fare'])
    df['Age_x_Class'] = df['Age'] * df['Pclass']
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df['Is_Child'] = (df['Age'] <= 12).astype(int)
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_Size', 'Is_Alone',
                'Fare_Per_Person', 'Age_Group', 'Sex_Encoded', 'Title_Encoded',
                'Embarked_Encoded', 'Avg_Fare_Pclass', 'Avg_Age_Pclass',
                'Avg_Survival_Title', 'Age_Std_Pclass', 'Fare_Std_Pclass',
                'Fare_Log', 'Age_x_Class', 'Has_Cabin', 'Is_Child']
    result = df[features + ['Survived']].fillna(0)
    assert result.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in result.dtypes)
    print('Liczba cech:', len(features), '| Wymiary:', result.shape)
    print(result.head(10).round(3).to_string(index=False))
    print('Uwaga: Avg_Survival_Title korzysta z targetu i wymaga kodowania out-of-fold przy trenowaniu.')
