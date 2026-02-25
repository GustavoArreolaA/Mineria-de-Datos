import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData_Limpio.csv', encoding='latin-1')

df['date'] = pd.to_datetime(df['date'])

print(f"Inicio: {df['date'].min()}")
print(f"Fin: {df['date'].max()}")
print(f"Duración: {df['date'].max() - df['date'].min()}")


df['month'] = df['date'].dt.month
print(f"Mes con más registros (Moda): {df['month'].mode()[0]}")

print(df.describe())

print(df[['seasons_Spring', 'seasons_Summer', 'seasons_Autumn', 'seasons_Winter']].mean() * 100)

print(df.groupby('hour')['rented_bike_count'].sum().sort_values(ascending=False))

print(df.corr(numeric_only=True))

print(df.skew(numeric_only=True))
print(df.kurtosis(numeric_only=True))