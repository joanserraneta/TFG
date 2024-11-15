import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Extraer las coordenadas x y j del archivo
coordinates_xy = []
with open("MSS_14456_6_0005_0001_1_0184.idx", "r") as file:
    for line in file:
        elements = line.split()
        i=0
        for el in elements:
            
            print("num:", i,"  ", el)
            i= i+1
        coordinates_xy.append([float(elements[3]), float(elements[4])])  # Coordenadas x y 

Xli_xy = preprocessing.normalize(coordinates_xy)


kmeans_algorithm = KMeans(
    n_clusters=20,  # Ajusta según el número de grupos esperados
    init='k-means++',
    n_init=10,
    max_iter=300,
    tol=0.0001,
    random_state=111,
    algorithm='elkan'
)
kmeans_algorithm.fit(Xli_xy)

# Obtener etiquetas y centroides
labels_xy = kmeans_algorithm.labels_
centroids_xy = kmeans_algorithm.cluster_centers_

# Crear la gráfica
plt.figure(figsize=(12, 8))
plt.scatter(Xli_xy[:, 0], Xli_xy[:, 1], c=labels_xy, cmap='viridis', marker='o', label="Puntos de datos")
plt.scatter(centroids_xy[:, 0], centroids_xy[:, 1], c='red', marker='x', s=150, label="Centroides")
plt.title("Agrupación de coordenadas x-y usando KMeans")
plt.xlabel("Coordenada x (normalizada)")
plt.ylabel("Coordenada y (normalizada)")
plt.legend()
plt.show()
