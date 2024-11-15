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
df.sort_values(by=['y'])
df['unos'] = 1
#xnorm = preprocessing.normalize(df[['x','y']])

xnorm = df[['x']]

xkmeans_algorithm = KMeans(
    n_clusters=20,  # Ajusta según el número de grupos esperados
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm='elkan'
)

xkmeans_algorithm.fit(xnorm)


# xkmeans_algorithm.fit(xnorm)
labels_x = xkmeans_algorithm.labels_
centroids_x = xkmeans_algorithm.cluster_centers_

df['label'] = labels_x

# Crear la gráfica
plt.figure(figsize=(12, 8))
plt.scatter(xnorm,df['unos'], c=labels_x, cmap='viridis', marker='o', label="Puntos de datos")
#plt.scatter(centroids_x[:,0], centroids_x[:,1], c='red', marker='x', s=150, label="Centroides")
plt.title("Agrupación de coordenadas x-y usando KMeans")
plt.xlabel("Coordenada x (normalizada)")
plt.ylabel("Coordenada y (normalizada)")
plt.legend()
plt.show()


filtrat = df[df['label'] == 1]

print (filtrat)