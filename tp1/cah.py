import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_rand_score

# Reuse data from ex1

mean1 = np.array([2, 2])
cov1 = np.array([[2, 0], [0, 2]])  # 2 * Identity Matrix
class1_data = np.random.multivariate_normal(mean1, cov1, 256)

mean2 = np.array([-4, -4])
cov2 = np.array([[6, 0], [0, 6]])  # 6 * Identity Matrix
class2_data = np.random.multivariate_normal(mean2, cov2, 256)

data = np.concatenate([class1_data, class2_data])
true_labels = np.array([0] * 256 + [1] * 256)


# ex3

Z_complete = linkage(data, method='complete', metric='euclidean')

plt.title("CAH")
d = dendrogram(Z_complete, color_threshold=0)
#plt.show()

# q3 ?
K=3
seuil_manuel=np.float64(14)
groupes_cah_manuel = fcluster(Z_complete, t=seuil_manuel, criterion='distance')
plt.figure(figsize=(10,7))
plt.title(f"CAH - Dendrogramme pour K = {K} clusters")
d = dendrogram(Z_complete, color_threshold=K)
plt.axhline(y=seuil_manuel, c='grey', lw=1, linestyle='dashed')
#plt.show()

# q4

diff = Z_complete[-2, 2] - Z_complete[-3, 2]
print(f"diff : {diff}")

seuil_auto = (Z_complete[-2, 2] + Z_complete[-3, 2]) / 2
print(f"Seuil : {seuil_auto}")

plt.figure(figsize=(10,7))
plt.title("CAH - Dendrogramme avec un seuil automatique")
dendrogram(Z_complete, seuil_auto)
plt.axhline(y=seuil_auto, c='grey', lw=1, linestyle='dashed')
plt.show()

groupes_cah = fcluster(Z_complete, t=seuil_auto, criterion='distance')
print("Cluster labels generated: ", np.unique(groupes_cah))

methods = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(15, 10))

for i, method in enumerate(methods, 1):
    # 1. Compute Linkage
    Z = linkage(data, method=method, metric='euclidean')
    
    # 2. Get labels for K=2 (to compare with our True Labels from Exercise 1)
    # We grab the last distances to automate the cut for K=2
    # Cut just below the final merge
    t_k2 = (Z[-1, 2] + Z[-2, 2]) / 2
    labels_k2 = fcluster(Z, t=t_k2, criterion='distance')
    
    # 3. Calculate Accuracy (ARI)
    # Note: We are comparing to 'true_labels' from Exercise 1
    ari = adjusted_rand_score(true_labels, labels_k2)
    
    # 4. Plot Dendrogram
    plt.subplot(2, 2, i)
    plt.title(f"Method: {method.upper()} | ARI (K=2): {ari:.3f}")
    dendrogram(Z, color_threshold=t_k2)
    plt.xlabel("Index")
    plt.ylabel("Distance")

plt.tight_layout()
plt.show()


