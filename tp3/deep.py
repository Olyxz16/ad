import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


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

loss = history_profond.history['loss']
val_loss = history_profond.history['val_loss']
epochs_range = range(50)

plt.figure(figsize=(10, 6))

plt.plot(epochs_range, loss, 'b', label='Training loss')

plt.plot(epochs_range, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation Loss (Modèle Profond)')
plt.xlabel('Epochs')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend()
plt.grid(True)
plt.show()
