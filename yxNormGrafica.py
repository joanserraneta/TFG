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

xnorm = preprocessing.normalize(df[['x']])
ynorm = preprocessing.normalize(df[['y']])

xkmeans_algorithm = KMeans(
    n_clusters=20,  # Ajusta según el número de grupos esperados
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm='elkan'
)

xkmeans_algorithm.fit(xnorm, ynorm)


# xkmeans_algorithm.fit(xnorm)
labels_x = xkmeans_algorithm.labels_
centroids_x = xkmeans_algorithm.cluster_centers_


# ykmeans_algorithm = KMeans(
#     n_clusters=20,  # Ajusta según el número de grupos esperados
#     init='k-means++',
#     n_init=10,
#     max_iter=300,
#     tol=0.0001,
#     random_state=111,
#     algorithm='elkan'
# )
# ykmeans_algorithm.fit(ynorm)
# labels_y = ykmeans_algorithm.labels_
# centroids_y = ykmeans_algorithm.cluster_centers_


# Crear la gráfica
plt.figure(figsize=(12, 8))
plt.scatter(xnorm, ynorm, c=labels_x, cmap='viridis', marker='o', label="Puntos de datos")
plt.scatter(centroids_x[:,0], centroids_x[:,1], c='red', marker='x', s=150, label="Centroides")
plt.title("Agrupación de coordenadas x-y usando KMeans")
plt.xlabel("Coordenada x (normalizada)")
plt.ylabel("Coordenada y (normalizada)")
plt.legend()
plt.show()
