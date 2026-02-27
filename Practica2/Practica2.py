import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData_Limpio.csv', encoding='latin-1')

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
print(f"Mes con más registros (Moda): {df['month'].mode()[0]}")

print(df.describe())

print(df[['seasons_Spring', 'seasons_Summer', 'seasons_Autumn', 'seasons_Winter']].mean() * 100)

print(df.groupby('hour')['rented_bike_count'].sum().sort_values(ascending=False))

print(df.corr(numeric_only=True))

#Asimetria
print(df.skew(numeric_only=True))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.histplot(df['rented_bike_count'], kde=True, color="skyblue")

plt.axvline(df['rented_bike_count'].mean(), color='red', linestyle='--', label=f"Media: {df['rented_bike_count'].mean():.2f}")
plt.axvline(df['rented_bike_count'].median(), color='green', linestyle='-', label=f"Mediana: {df['rented_bike_count'].median():.2f}")

plt.title(f"Distribución de Rentas\nAsimetría: {df['rented_bike_count'].skew():.2f} | Curtosis: {df['rented_bike_count'].kurtosis():.2f}")
plt.legend()
plt.show()

#Curtosis
print(df.kurtosis(numeric_only=True))

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['rented_bike_count'], color="lightgreen")

plt.title("Boxplot de Rentas (Detección de Outliers y Asimetría)")
plt.xlabel("Número de Bicicletas Rentadas")
plt.show()