import numpy as np
import matplotlib.pyplot as plt

# data generation

N = 128

mean1 = [2, 2]
cov1 = [[2,0],[0,2]]
data1 = np.random.multivariate_normal(mean1, cov1, N)

mean2 = [-4, -4]
cov2 = [[6,0],[0,6]]
data2 = np.random.multivariate_normal(mean2, cov2, N)

data = np.concatenate([data1, data2])

# coalesce

def coalescence(x, K, g):
    """
    x: Data matrix (N x 2)
    K: Number of clusters
    g: Initial centers (K x 2)
    
    Returns:
    clas: Vector of labels (N,)
    g2: Final centers (K x 2)
    """

    x = np.array(x, dtype=np.float64)
    g = np.array(g, dtype=np.float64)

    max_iter = 100
    centers = np.array(g, dtype=np.float64)
    
    global clas

    for i in range(max_iter):
        distances = np.zeros((x.shape[0], K))
        
        for k in range(K):
            dist_vec = np.sum((x - centers[k])**2, axis=1)
            distances[:, k] = dist_vec
            
        clas = np.argmin(distances, axis=1)
        
        new_centers = np.zeros_like(centers)
        
        for k in range(K):
            points_in_cluster = x[clas == k]
            
            if len(points_in_cluster) > 0:
                new_centers[k] = np.mean(points_in_cluster, axis=0)
            else:
                new_centers[k] = centers[k]
        
        if np.allclose(centers, new_centers, atol=1e-4):
            break
            
        centers = new_centers
        
    return clas, centers

# q2

K = 2

random_indices = np.random.choice(data.shape[0], K, replace=False)
initial_centers = data[random_indices]

labels, final_centers = coalescence(data, K, initial_centers)

# q3

plt.figure(figsize=(8, 6))

# Plot the points, colored by the labels found by our function
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6, label='Data Points')

# Plot the INITIAL centers (Small Red X)
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='red', marker='x', s=100, label='Initial Centers')

# Plot the FINAL centers (Big Red Circles)
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='o', s=200, edgecolors='black', label='Final Centers')

plt.title('My "Coalescence" (K-Means) Result')
plt.legend()
plt.grid(True)
plt.show()
