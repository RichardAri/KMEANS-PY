import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
import seaborn as sb
from sklearn.model_selection import KFold

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
df = pd.read_csv('/content/drive/My Drive/analisis.csv')

df.head()

print(df.groupby('categoria').size())
df.drop("categoria", axis=1).hist()
plt.show()

sb.pairplot(df.dropna(),hue='categoria',size=4, vars=['op','ex','ag'], kind='scatter')

x = np.array(df[['op','ex','ag']])
y = np.array(df['categoria'])

fig = plt.figure()
ax = Axes3D(fig)
colores = ['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple']
asignar = []
for row in y:
    asignar.append(colores[row])
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=asignar,s=60)

# metodo del Codo
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]

plt.plot(Nc, score)
plt.xlabel('Número de Clusters')
plt.ylabel('Puntuación')
plt.title('Curva de Codo (Elbow Curve)')
plt.show()

# indice de silueta
silhouette_scores = []
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(x)
    label = kmeans.labels_
    sil_score = silhouette_score(x, label, metric='euclidean')
    silhouette_scores.append(sil_score)

plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Número de Clusters')
plt.ylabel('Índice de Silueta')
plt.title('Índice de Silueta para K-means')
plt.show()

# validacion cruzada de kmeans
def kmeans_cross_validation(data, n_clusters, n_splits=5):
    kf = KFold(n_splits=n_splits)
    scores = []
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        kmeans = KMeans(n_clusters=n_clusters).fit(train_data)
        labels_train = kmeans.predict(train_data)
        labels_test = kmeans.predict(test_data)
        score = silhouette_score(test_data, labels_test, metric='euclidean')
        scores.append(score)
    return np.mean(scores)

cross_val_scores = [kmeans_cross_validation(x, n_clusters=i) for i in range(2, 11)]

plt.plot(range(2, 11), cross_val_scores)
plt.xlabel('Número de Clusters')
plt.ylabel('Puntuación de Validación Cruzada')
plt.title('Validación Cruzada para K-means')
plt.show()

# pureza del cluster
def cluster_purity(labels_true, labels_pred):
    matrix = confusion_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

kmeans = KMeans(n_clusters=5).fit(x)
labels_pred = kmeans.labels_
purity = cluster_purity(y, labels_pred)
print(f'Pureza del Clúster: {purity:.2f}')
