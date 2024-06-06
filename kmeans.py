import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import seaborn as sb

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

#import de el dataset 
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
df = pd.read_csv('/content/drive/My Drive/analisis.csv')

df.head()

print(df.groupby('categoria').size())
#df.drop(["categoria"],1).hist()
df.drop("categoria", axis=1).hist() #histograma para todos menos 'categoria'
plt.show()

# creamos un pairplot de el dataset coloreado por categoria
sb.pairplot(df.dropna(),hue='categoria',size=4, vars=['op','ex','ag'], kind='scatter')

x = np.array(df[['op','ex','ag']])
y = np.array(df['categoria'])

# creamos un grafico de dispersion en 3D de las caracteristicas coloreadas por categoría
fig = plt.figure()
ax = Axes3D(fig)
colores = ['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
asignar = []
for row in y:
    asignar.append(colores[row])
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=asignar,s=60)

# determina el numero optimo de clusters usando el metodo de elbow
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
score

# Mostramos la curva de elbow 
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve') # Elbow Method
plt.show()

kmeans 