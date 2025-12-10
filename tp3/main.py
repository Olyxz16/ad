import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

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

"""
On peut voir qu'on a effectivement un réseau un réseau a trois couches, respectivement de taille 784, 32, 784.
On peut également voir que le réseau a 50,992 paramètres en tout, ce qui correspond à 199.19KB en mémoire.
La première couche est de type InputLayer et les deux autres sont des couches Dense.
"""

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
nepochs = 20
autoencoder_train = autoencoder.fit(x_train, x_train, epochs=nepochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(nepochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Losses')
plt.legend()
plt.show()

"""
On voit d'après le graphe que le modèle suit plutôt la courbe décrite par les données de test.
"""

encoded_imgs = encoder.predict(x_test)
decoded_img = decoder.predict(encoded_imgs)
