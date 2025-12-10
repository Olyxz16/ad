import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

"""
DONNEES
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

"""
SIMPLE
"""

taille_code = 32
input_img = keras.Input(shape=(784,))
encoded = layers.Dense(taille_code, activation='relu')(input_img)
decoded = layers.Dense(784, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)
encoded_input = keras.Input(shape=(taille_code,))
decoded_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoded_layer(encoded_input))

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
nepochs = 20
autoencoder_train = autoencoder.fit(x_train, x_train, epochs=nepochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

"""
PROFOND
"""

input_img = keras.Input(shape=(784,))

encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

autoencoder_profond = keras.Model(input_img, decoded)

autoencoder_profond.summary()

autoencoder_profond.compile(optimizer='adam', loss='binary_crossentropy')

history_profond = autoencoder_profond.fit(x_train, x_train,
                                          epochs=50,
                                          batch_size=256,
                                          shuffle=True,
                                          validation_data=(x_test, x_test))

decoded_imgs_profond = autoencoder_profond.predict(x_test)

# 2. Calcul des métriques pour le modèle profond (pour comparer les chiffres)
mse_profond = np.mean([mse(x_test[i], decoded_imgs_profond[i]) for i in range(len(x_test))])
ssim_profond = np.mean([ssim(x_test[i].reshape(28,28), 
                             decoded_imgs_profond[i].reshape(28,28), 
                             data_range=1.0) for i in range(len(x_test))])

print(f"--- RÉSULTATS COMPARATIFS ---")

# 3. Visualisation : Original vs Simple vs Profond
n = 8  # Nombre d'images à afficher
plt.figure(figsize=(20, 6))

for i in range(n):
    # A. Image Originale
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("Original", fontsize=14, loc='left')

    # B. Reconstruction - Modèle Simple (1 couche cachée)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("Auto-encodeur Simple", fontsize=14, loc='left')

    # C. Reconstruction - Modèle Profond (5 couches cachées)
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs_profond[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("Auto-encodeur Profond", fontsize=14, loc='left')

plt.show()
