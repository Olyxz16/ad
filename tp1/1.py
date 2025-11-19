import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Class 1: N((2,2), 2*I)
mean1 = np.array([2, 2])
cov1 = np.array([[2, 0], [0, 2]])  # 2 * Identity Matrix
class1_data = np.random.multivariate_normal(mean1, cov1, 256)

# Class 2: N((-4,-4), 6*I)
mean2 = np.array([-4, -4])
cov2 = np.array([[6, 0], [0, 6]])  # 6 * Identity Matrix
class2_data = np.random.multivariate_normal(mean2, cov2, 256)

# Plotting
plt.plot(class1_data[:,0], class1_data[:,1], 'o', color='blue', label='Classe 1: N([2,2], 2I)')
plt.plot(class2_data[:,0], class2_data[:,1], 'x', color='red', label='Classe 2: N([-4,-4], 6I)')

plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
