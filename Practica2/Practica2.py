import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData_Limpio.csv', encoding='latin-1')



print(df.describe())

print(df.skew(numeric_only=True))
print(df.kurtosis(numeric_only=True))

print(df[['seasons_Spring', 'seasons_Summer', 'seasons_Autumn', 'seasons_Winter']].mean() * 100)
print(df['hour'].value_counts())
print(df.groupby('hour')['rented_bike_count'].sum().sort_values(ascending=False))

print(df.corr(numeric_only=True))