import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples


# Reuse data from ex1

mean1 = np.array([2, 2])
cov1 = np.array([[2, 0], [0, 2]])  # 2 * Identity Matrix
class1_data = np.random.multivariate_normal(mean1, cov1, 256)

mean2 = np.array([-4, -4])
cov2 = np.array([[6, 0], [0, 6]])  # 6 * Identity Matrix
class2_data = np.random.multivariate_normal(mean2, cov2, 256)

data = np.concatenate([class1_data, class2_data])
true_labels = np.array([0] * 256 + [1] * 256)


# ex2

K_values = [2, 3, 4, 5, 6]
silhouette_scores = []
inertias = []

for k in K_values:
    kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++')
    kmeans.fit(data)
    labels = kmeans.labels_

    score = silhouette_score(data, labels)
    silhouette_scores.append(score)
    
    inertias.append(kmeans.inertia_)

fig, ax1 = plt.subplots()
ax1.plot(K_values, inertias)
ax1.set_ylabel('Inertia')
ax1.tick_params('both', color="blue")

ax2 = ax1.twinx()
ax2.plot(K_values, silhouette_scores)
ax2.set_ylabel('Silhouette score')
ax2.tick_params(axis='both', color='tab:orange')

ax1.set_xlabel("Nombre de clusters (K)")
ax1.set_title("Differences between cluster inertia and silhouette scores")
ax1.grid(True)
fig.tight_layout()
plt.show()

"""
Meilleur paramètre K : 2, car la silouette est au maximum
"""
