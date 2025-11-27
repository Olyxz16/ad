import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

data_temperature = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)
n = len(data_temperature)

data = data_temperature.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])

data_scaled = scale(data)
matrice_distances_condensed = pdist(data_scaled, metric='euclidean')
square_matrix = pd.DataFrame(
    squareform(matrice_distances_condensed),
    index=data.index,
    columns=data.index
)

print("Matrice de dissimilarité : ")
print(square_matrix.iloc[:, :].round(2))

Z = linkage(matrice_distances_condensed, method='ward', metric='euclidean')

plt.figure(figsize=(12,6))
plt.title("CAH dendrogram")

seuil = 7

dendrogram(Z, labels=list(data.index), leaf_rotation=45, color_threshold=seuil)

plt.axhline(y=seuil, c='grey', lw=1, linestyle='dashed')
plt.show()

# q3

labels_cah = fcluster(Z, t=seuil, criterion='distance')
data['Cluster_CAH'] = labels_cah

print("Number of cities per cluster")
print(data['Cluster_CAH'].value_counts().sort_index())

for i in sorted(data['Cluster_CAH'].unique()):
    cities_in_cluster = data.index[data['Cluster_CAH'] == i].tolist()
    print(f"Cluster {i}: {cities_in_cluster}")

# q4

Z_avg = linkage(data_scaled, method='average', metric='euclidean') 
seuil_avg = 3.5
labels_avg = fcluster(Z_avg, t=seuil_avg, criterion='distance')

Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values
plt.scatter(Coord[:, 1], Coord[:, 0], c=labels_avg, s=20, cmap='viridis')

nom_ville = list(data_temperature.index)

for i, txt in enumerate(nom_ville):
    plt.annotate(txt, (Coord[i,1], Coord[i,0]))

plt.show()

# 2.3 
# q1

K_range = range(2, 10)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)

# q2

plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel("Nombre de clusters K")
plt.ylabel("Inerties")
plt.title("Règle du coude")
plt.grid(True)
plt.show()

# q3

K_optimal = 3
kmeans_final = KMeans(n_clusters=K_optimal, n_init=10)
kmeans_final.fit(data_scaled)

data['Cluster_KMeans'] = kmeans_final.labels_

Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values

plt.figure(figsize=(8,8))
plt.title(f"Partion KMeans (K={K_optimal}) sur carte")

plt.scatter(Coord[:, 1], Coord[:, 0], c=kmeans_final.labels_, s=20, cmap='viridis')

for i, txt in enumerate(data_temperature.index):
    plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# 2.4

# q1

# Utilisation de table de contingence

comparison = pd.crosstab(
    data['Cluster_CAH'],
    data['Cluster_KMeans'],
    rownames=['CAH Cluster'],
    colnames=['KMeans Cluster']
)

print("Table de comparaison (CAH vs K-Means) :")
print(comparison)
