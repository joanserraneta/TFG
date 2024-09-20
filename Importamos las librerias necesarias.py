# Importamos las librerias necesarias
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
py.offline.init_notebook_mode(connected = True)
pio.renderers.default='browser'

df = pd.read_csv('/content/Mall_Customers.csv')
print('Dimensiones del df:', df.shape)
#____________________________________________________________
# Vamos a sacar informacion sobre los tipos y nulos del df
df.drop_duplicates(inplace = True)
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'tipo de la columna'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'campos nulos (cant)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                         rename(index={0:'campos nulos (%)'}))
display(tab_info)
#__________________
#
display(df[:5])

df['Gender'] = pd.get_dummies(df['Gender']).values[:,0]
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))

X = df.copy()
X.drop(labels=['CustomerID'], axis=1, inplace=True)
X1 = preprocessing.normalize(X)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12))
visualizer.fit(X1)        # Entrenamos con los datos
visualizer.show()        # Renderizamos la imagen

algorithm = KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
algorithm.fit(X1)
labels = algorithm.labels_
centroids = algorithm.cluster_centers_
df['label'] =  labels

fig = px.scatter_3d(df, x='Age', y='Spending Score (1-100)', z='Annual Income (k$)',
              color='label')
fig.show()
fig.write_html("/content/file.html")

