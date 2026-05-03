import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('SeoulBikeData_Limpio.csv')


df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['date']) 
df = df.sort_values('date')

# Crear el índice de tiempo
df['time_index'] = np.arange(len(df))

# Extraer características 
df['hour'] = df['date'].dt.hour
df['month'] = df['date'].dt.month

# Definir variables
features = ['time_index', 'temperature', 'hour', 'humidity']
X = df[features]
y = df['rented_bike_count']

# División Entrenamiento y Prueba (80/20)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Entrenar el modelo
modelo_forecast = LinearRegression()
modelo_forecast.fit(X_train, y_train)

# --- Predicción de Nuevos Datos---

ultimo_indice = df['time_index'].max()
nueva_data = pd.DataFrame([[ultimo_indice + 1, 20.5, 14, 50]], columns=features) #Dataset nuevo para evitar errores

prediccion_futura = modelo_forecast.predict(nueva_data)
print(f"Predicción para la siguiente hora: {max(0, prediccion_futura[0]):.2f} bicicletas")

plt.title('Pronóstico de Renta de Bicicletas', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Tiempo (Muestra de Horas)', fontsize=12)
plt.ylabel('Bicicletas Rentadas', fontsize=12)
plt.plot(y_test.values[:100], label='Real')
plt.plot(modelo_forecast.predict(X_test)[:100], label='Predicción')
plt.legend()
plt.show()