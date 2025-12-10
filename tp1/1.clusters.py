import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Class 1: N((2,2), 2*I)
mean1 = np.array([2, 2])
cov1 = np.array([[2, 0], [0, 2]])  # 2 * Identity Matrix
class1_data = np.random.multivariate_normal(mean1, cov1, 256)

# Class 2: N((-4,-4), 6*I)
mean2 = np.array([-4, -4])
cov2 = np.array([[6, 0], [0, 6]])  # 6 * Identity Matrix
class2_data = np.random.multivariate_normal(mean2, cov2, 256)

data = np.concatenate([class1_data, class2_data])
true_labels = np.array([0] * 256 + [1] * 256)

K = 2
kmeans = KMeans(n_clusters=K, n_init=10, init='k-means++')
km = kmeans.fit(data)
if km.labels_ is None:
    exit(0)

ari_score = adjusted_rand_score(true_labels, kmeans.labels_)
print(f"Adjusted Rand Score: {ari_score:.4f}")

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.scatter(data[:,0], data[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title(f'K-Means Result (K={K}')
plt.xlabel('X dimension')
plt.ylabel('Y dimension')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis')
plt.title('True classification')

plt.show()
