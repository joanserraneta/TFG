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

    #el resto de los elementos en un array np
    nFila = np.array(elementos[1:], dtype=float)
    
    # Apilar la nueva fila numérica al array
    arr = np.vstack([arr, nFila])
    
    # Agregar el primer elemento a la lista de primeros elementos
    palabras.append(word)

df = pd.DataFrame(arr, index=palabras, columns=list("npxyjz"))
df.sort_values(by=['y'])
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

############################# 20 linies ###########################################


# Función para calcular distorsión (inertia en sklearn)
def calcular_distorsion(data, max_k, umbral_distorsion):
    distorsiones = []
    k_valores = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distorsion = kmeans.inertia_
        distorsiones.append(distorsion)
        k_valores.append(k)
        
        # Verificar si la distorsión es menor que el umbral
        if distorsion < umbral_distorsion:
            print(f"K óptimo con distorsión < {umbral_distorsion}: {k}")
            break
    
    print( k_valores, distorsiones)


lines = df[['y', 'z']]
columns = df[['x', 'j']]

#elbow de lineas
Xli = preprocessing.normalize(lines)
model = KMeans()
visualizer= KElbowVisualizer(model, k=(10,25))
visualizer.fit(lines)
visualizer.show()
print(visualizer.elbow_value_, visualizer.estimator)

#- kmeans++ intenta inicializar los centroides de manera inteligente 
#- n_init = Se iniciará 10 veces ocn distintos centroides y se seleccionara el mejor (para evitar minimos locales)
#- Tol para cuando los centroides se muevan menos de esta distancia se parará 
# -Algorithm='elkan': Algoritmo subyacente 'elkan', versión más rápida de K-Means que utiliza ciertas optimizaciones geométricas. 
#  para conjuntos de datos de baja dimensionalidad y puede acelerar el proceso de agrupación.

algorithm = KMeans(n_clusters = 20 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
algorithm.fit(Xli)
labels = algorithm.labels_ # etiquetas de clusters (K)
centroids = algorithm.cluster_centers_

# df['label'] = labels: Aquí estás añadiendo una nueva columna al DataFrame df llamada 'label', que contiene las etiquetas de los clusters para cada fila de datos. 
# Esto te permite saber a qué cluster pertenece cada punto de datos en el DataFrame original.

df['label'] =  labels

#Grafica xula de firefox 
#fig = px.scatter(df, x="x", y="y")
#fig.show()


#df.drop(['x'], 1).hist()


# Inicializar un diccionario para almacenar DataFrames por label
dfs_by_label = {}

# Iterar sobre los valores únicos de 'label'
for label in df['label'].unique():
    # Filtrar el DataFrame por el valor de 'label'
    filtered_df = df[df['label'] == label]
    
    # Almacenar el DataFrame en el diccionario
    dfs_by_label[label] = filtered_df


# Mostrar los DataFrames creados
for label, dataframe in dfs_by_label.items():
    print(f"\nDataFrame for label {label}:\n{dataframe}")

    columns = dataframe[['x']]

    Xli = preprocessing.normalize(columns)
    model = KMeans()
    visualizer= KElbowVisualizer(model, k=(1,8))
    visualizer.fit(lines)
    #visualizer.show()


