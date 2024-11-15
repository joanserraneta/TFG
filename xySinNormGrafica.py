import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd 
import numpy as np

# Extraer las coordenadas x y j del archivo
f = open("MSS_14456_6_0005_0001_1_0184.idx", "r")
arr = np.empty((0, 6))
palabras = []

for line in f:

    elementos = line.split()
    word = elementos[0]

    #el resto de los elementos en un array np
    nFila = np.array(elementos[1:], dtype=float)
    
    # Apilar la nueva fila numérica al array
    arr = np.vstack([arr, nFila])
    
    # Agregar el primer elemento a la lista de primeros elementos
    palabras.append(word)

df = pd.DataFrame(arr, index=palabras, columns=list("npxyjz"))


kmeans_algorithm = KMeans(
    n_clusters=20,  # Ajusta según el número de grupos esperados
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm='elkan'
)
kmeans_algorithm.fit(df[['x', 'y']])

# Obtener etiquetas y centroides
labels_xy = kmeans_algorithm.labels_
centroids_xy = kmeans_algorithm.cluster_centers_

# Crear la gráfica
plt.figure(figsize=(12, 8))
plt.scatter(df['x'], df[['y']], c=labels_xy, cmap='viridis', marker='o', label="Puntos de datos")
plt.scatter(centroids_xy[:, 0], centroids_xy[:, 1], c='red', marker='x', s=150, label="Centroides")
plt.title("Agrupación de coordenadas x-y usando KMeans")
plt.xlabel("Coordenada x (normalizada)")
plt.ylabel("Coordenada y (normalizada)")
plt.legend()
plt.show()
