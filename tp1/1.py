import numpy as np
import matplotlib.pyplot as plt
import random
import os

mean = np.array([0,0])
cov = np.array([[1,0.5],[0.5,1]])
X = (np.random.multivariate_normal(mean, cov, 256))

plt.plot(X[:,0],X[:,1], 'o', label='Individu')
plt.legend()
plt.show()
