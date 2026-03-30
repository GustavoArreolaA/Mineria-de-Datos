import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


df = pd.read_csv('SeoulBikeData_Limpio.csv')
#Convertir "rented_bike_count" en categorías
# 0 = baja, 1 = media, 2 = alta
df['demanda'] = pd.qcut(df['rented_bike_count'], q=3, labels=[0, 1, 2])

X = df.drop(columns=['date', 'cluster', 'rented_bike_count', 'demanda'])
y = df['demanda']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

accuracies = []
k_values = range(1, 11)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

best_k = k_values[accuracies.index(max(accuracies))]
print("Mejor K:", best_k)
print("Accuracy:", round(max(accuracies), 4))

plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs K')
plt.xlabel('Número de vecinos (K)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='viridis', s=40, alpha=0.6)

cbar = plt.colorbar(scatter, ticks=[0,1,2])
cbar.ax.set_yticklabels(['Baja demanda', 'Media demanda', 'Alta demanda'])


plt.title('KNN - Clasificación de demanda de bicicletas')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()

