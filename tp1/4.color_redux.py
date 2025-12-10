import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

# q1

img = np.float32(mpimg.imread('visage.bmp'))
reshaped_img = img.reshape(256 * 256, 3)

K = 2
kmeans = KMeans(n_clusters=K, n_init=10, init='k-means++').fit(reshaped_img)
if kmeans.labels_ is None:
    exit(0)

labels = kmeans.predict(reshaped_img)
centers = kmeans.cluster_centers_

pixels = centers[labels]
pixels = np.reshape(pixels, (256, 256, 3))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image (Thousands of colors)')
plt.imshow(img/255)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Compressed Image ({K} colors)')
plt.imshow(pixels/255)
plt.axis('off')

plt.show()

# q2
# dans les notes
