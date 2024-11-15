import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.io as pio
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings("ignore")


f = open("MSS_14456_6_0005_0001_1_0184.idx", "r")
arr = np.empty((0, 6))
palabras = []

for line in f:
    elementos = line.split()
    word = elementos[0]
    nFila = np.array(elementos[1:], dtype=float)
    arr = np.vstack([arr, nFila])
    palabras.append(word)

df = pd.DataFrame(arr, index=palabras, columns=list("npxyjz"))

# print(df['x'].max())

plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (25 , 6))
n = 0 

#Agrupacions per linies  
for x in ['y']:
    n += 1
    plt.subplot(1 , 1 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.histplot(df[x] , bins = 100, kde=False) #nom fila i numero de contenedors del histograma
    plt.title('Distplot of {}'.format(x))
    
plt.show()

lines = df[['y', 'z']]
columns = df[['x', 'j']]

lineaB = df[['y']]

#elbow de lineas
Xli = preprocessing.normalize(lineaB)
model = KMeans()
visualizer= KElbowVisualizer(model, k=(10,25))
visualizer.fit(Xli)
visualizer.show()
print(visualizer.elbow_value_, visualizer.estimator)




algorithm = KMeans(n_clusters = 20 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
algorithm.fit(Xli)
labels = algorithm.labels_ # etiquetas de clusters (K)
centroids = algorithm.cluster_centers_

# df['label'] = labels: nueva columna al DataFrame df llamada 'label', que contiene las etiquetas de los clusters para cada fila de datos. 

df['label'] =  labels



dfs_by_label = {}
for label in df['label'].unique():
    # Filtra
    filtered_df = df[df['label'] == label]
    
    # Almacenar el DataFrame en el diccionario
    dfs_by_label[label] = filtered_df


# Mostrar los DataFrames creados
for label, dataframe in dfs_by_label.items():
    if label == 2:
        print(f"\nDataFrame for label {label}:\n{dataframe}")

    columns = dataframe[['x']]

    Xcol = preprocessing.normalize(columns)
    model = KMeans()
    visualizer= KElbowVisualizer(model, k=(1,8))
    visualizer.fit(Xcol)
    #visualizer.show()


