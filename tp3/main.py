import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse


def tester_autoencodeur(taille_code, nb_epochs, x_train, x_test):
    """
    Entraîne un modèle avec des paramètres donnés et retourne les métriques.
    """
    input_dim = 784
    
    input_img = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(taille_code, activation='relu')(input_img)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = keras.Model(input_img, decoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    history = autoencoder.fit(x_train, x_train, epochs=nb_epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test), verbose=0)
    
    decoded_imgs = autoencoder.predict(x_test)
    
    score_mse = np.mean([mse(x_test[i], decoded_imgs[i]) for i in range(len(x_test))])
    
    score_ssim = np.mean([ssim(x_test[i].reshape(28,28), 
                               decoded_imgs[i].reshape(28,28), 
                               data_range=1.0) for i in range(len(x_test))])
    
    return score_mse, score_ssim, history.history['loss'][-1], history.history['val_loss'][-1]


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

n = 5
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n , i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

liste_mse = []
liste_ssim = []

for i in range(len(x_test)):
    img_originale = x_test[i].reshape(28,28)
    img_decode = decoded_img[i].reshape(28,28)

    valeur_mse = mse(x_test[i], decoded_img[i])
    liste_mse.append(valeur_mse)

    valeur_ssim = ssim(img_originale, img_decode, data_range=1.0)
    liste_ssim.append(valeur_ssim)

liste_mse = np.array(liste_mse)
liste_ssim = np.array(liste_ssim)

print(f"MSE Moyen : {np.mean(liste_mse)}")
print(f"SSIM Moyen : {np.mean(liste_ssim)}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(liste_mse)
plt.title('Distribution du MSE (Plus bas est mieux)')

plt.subplot(1, 2, 2)
plt.boxplot(liste_ssim)
plt.title('Distribution du SSIM (Plus haut est mieux)')

plt.show()

# 2.4

valeurs_code = [4,8,16,32,64]
resultats_mse = []
resultats_ssim = []


for t in valeurs_code:
    m, s, _, _ = tester_autoencodeur(taille_code=t, nb_epochs=20, x_train=x_train, x_test=x_test)
    resultats_mse.append(m)
    resultats_ssim.append(s)
    print(f"Taille: {t} -> MSE: {m:.4f}, SSIM: {s:.4f}")

# --- Visualisation ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(valeurs_code, resultats_mse, 'r-o')
plt.title('MSE en fonction de la taille du code')
plt.xlabel('Taille du code latent')
plt.ylabel('MSE (plus bas = mieux)')

plt.subplot(1, 2, 2)
plt.plot(valeurs_code, resultats_ssim, 'b-o')
plt.title('SSIM en fonction de la taille du code')
plt.xlabel('Taille du code latent')
plt.ylabel('SSIM (plus haut = mieux)')

plt.show()
