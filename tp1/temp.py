import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import scale

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

