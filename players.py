import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sb

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

#import de el dataset 
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
df = pd.read_csv('/content/drive/My Drive/SI_L09_KMEANS_DATASET.csv')


# seleccionamos las caracteristicas relevantes para el clustering
features = ['wins', 'kills', 'kdRatio', 'killstreak', 'level', 'losses', 'prestige', 
            'hits', 'timePlayed', 'headshots', 'averageTime', 'gamesPlayed', 
            'assists', 'misses', 'xp', 'scorePerMinute', 'shots', 'deaths']

df = df.dropna(subset=features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# determinamos el numero optimo de clusters usando el metodo del codo
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X_scaled).inertia_ for i in range(len(kmeans))]

# Mostramos la curva de elbow (metodo del codo)
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve') # Elbow Method
plt.show()

# elegimos el numero optimo de clusters basado en la curva del codo
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels

# visualizamos los clusters usando un pairplot
sb.pairplot(df, hue='cluster', vars=features[:4], palette='viridis')
plt.show()

# visualizar los clusters en 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, cmap='viridis', s=50)
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_xlabel('Wins')
ax.set_ylabel('Kills')
ax.set_zlabel('K/D Ratio')
plt.title('Grafico 3D de Clusters')
plt.show()

# numero de jugadores en cada cluster
cluster_counts = df['cluster'].value_counts().sort_index()
print(cluster_counts)
