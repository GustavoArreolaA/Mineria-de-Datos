import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('SeoulBikeData_Limpio.csv')

df.info()

X = df[['temperature']]
y = df[['rented_bike_count']]

modelo= LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)

r2 = r2_score(y, y_pred)
print(f"Puntaje R2: {r2:.3f}")

plt.figure(figsize=(10, 6))
sns.regplot(x=X, y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'#8B4513'}) # Línea café
plt.title(f'Modelo Lineal (R² = {r2:.3f})')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Rentas de Bicicletas')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.5, color='#87CEEB') # Tu azul claro
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2) # Línea de referencia
plt.title('Valores Reales vs Predicciones')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones del Modelo')
plt.show()