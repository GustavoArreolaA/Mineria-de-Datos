import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('SeoulBikeData_Limpio.csv')

#Seleccionar variables numéricas para agrupar
features = ['rented_bike_count', 'temperature', 'humidity', 'solar_radiation']
x = df[features]

#Escalar los datos 

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Método del Codo para hallar el número de clusters (k)
inertia = []
for k in range(1, 11):
    kmeans_prueba = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_prueba.fit(x_scaled)
    inertia.append(kmeans_prueba.inertia_)

#Coeficiente de silueta para determinar la mejor cantidad de grupos
mejor_score = -1
n_grupos = 2

for k in range(2, 11): 
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    
    print(f"Para k={k}, el Silhouette Score es: {score:.4f}")

    if score > mejor_score:
        mejor_score = score
        n_grupos = k

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.show()

modelo_final = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
df['cluster'] = modelo_final.fit_predict(x_scaled) + 1

pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=df['cluster'], palette='viridis')
plt.title("Visualización de Clusters usando PCA")
plt.show()
