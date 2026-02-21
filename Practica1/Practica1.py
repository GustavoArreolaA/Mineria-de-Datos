import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData.csv', encoding='latin-1')

print("Cantidad de filas y columnas antes de la limpieza")
print(df.shape)
print("\nCantidad de variables flotantes, enteras y categoricas antes de la limpieza")
df.info()

df.columns = [col.split('(')[0].strip().replace(' ', '_').lower() for col in df.columns]

print(df.head())

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

df.info()

df['holiday'] = df['holiday'].replace('No Holiday', '0')
df['holiday'] = df['holiday'].replace('Holiday', '1')
df['functioning_day'] = df['functioning_day'].replace('Yes', '1')
df['functioning_day'] = df['functioning_day'].replace('No', '0')

print(df.head())

df = pd.get_dummies(df, columns=['seasons'], drop_first=False)
df['holiday'] = df['holiday'].astype(int)
df['functioning_day'] = df['functioning_day'].astype(int)

print(df.head())
