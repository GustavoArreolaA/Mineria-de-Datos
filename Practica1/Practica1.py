import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData.csv', encoding='latin-1')

#Limpieza
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

#Clustering

df_cluster = df.drop(columns=['date'])
df_scaled = StandardScaler().fit_transform(df_cluster)

inercia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_init=10, n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inercia.append(kmeans.inertia_)

plt.plot(K_range, inercia, 'bx-')
plt.xlabel("Cantidad de Clusters")
plt.ylabel("Inercia")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_init=10, n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

print(df.groupby('cluster').mean(numeric_only=True))

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['cluster'], palette='viridis')
plt.title("Visualización de Clusters usando PCA")
plt.show()

#Prediccion

x = df.drop(columns=['rented_bike_count', 'date'])
y = df['rented_bike_count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(x_train, y_train)

y_pred = modelo_rf.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Evaluación del Modelo ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} bicicletas")
print(f"Coeficiente de Determinación (R2): {r2:.2f}")

# 6. Visualización de Predicciones vs Valores Reales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación entre valores reales y predicciones (Seoul Bike)')
plt.show()