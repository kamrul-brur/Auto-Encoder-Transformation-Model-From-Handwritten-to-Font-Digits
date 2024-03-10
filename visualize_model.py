import tensorflow as tf
from keras.utils import plot_model

# Assuming 'autoencoder' is your trained autoencoder model
autoencoder = tf.keras.models.load_model('modelh5/auto-encoder-model.h5', compile=False)

# Plot the model architecture to a file (PNG format)
plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True)
