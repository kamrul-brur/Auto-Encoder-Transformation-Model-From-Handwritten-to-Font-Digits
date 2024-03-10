from keras import layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam

def autoencoder_model():
    """
    Build and compile a convolutional autoencoder model.

    Returns:
    - keras.models.Model: Compiled autoencoder model.

    This function constructs a convolutional autoencoder model using Keras. The architecture consists of an
    encoder and a decoder with multiple convolutional layers. The model is compiled with the Adam optimizer
    and binary crossentropy loss.

    Example:
    # >>> autoencoder = autoencoder_model()
    # Returns a compiled autoencoder model.
    """
    input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder
    autoencoder = Model(input, x)
    optimizer = Adam(lr=0.001)
    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    autoencoder.summary()
    return autoencoder