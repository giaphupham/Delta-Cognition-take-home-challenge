import tensorflow as tf
from tensorflow.python.keras import layers, models

class Autoencoder:
    def __init__(self, input_dim=784):
        self.model = self.build_autoencoder(input_dim)

    def build_autoencoder(self, input_dim):
        input_img = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_img)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        return autoencoder

    def train(self, x_train, x_test, epochs=50, batch_size=256):
        self.model.fit(x_train, x_train, epochs=epochs, batch_size=batch_size,
                       shuffle=True, validation_data=(x_test, x_test))

    def evaluate(self, data):
        reconstruction = self.model.predict(data)
        reconstruction_error = tf.keras.losses.mean_squared_error(data, reconstruction)
        return reconstruction_error
