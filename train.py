import tensorflow as tf
from data_processor import load_dataset, display
from encoder_model import autoencoder_model
from config import Config as cfg


def train_model():
    print('MNIST dataset loading...')
    handwritten_train, synthetic_train, handwritten_validation, synthetic_validation, handwritten_test, synthetic_test = load_dataset()
    print('Dataset loading completed!!!')
    autoencoder = autoencoder_model()
    print('Handwritten Train Data : ',handwritten_train.shape)
    print('Synthetic Train Data : ', synthetic_train.shape)
    print('Handwritten Test Data : ', handwritten_test.shape)
    print('Synthetic Test Data : ', synthetic_test.shape)
    print('Handwritten Validation Data : ', handwritten_validation.shape)
    print('Synthetic Validation Data : ', synthetic_validation.shape,"\n")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    autoencoder.fit(
        x=handwritten_train,
        y=synthetic_train,
        epochs=cfg.epoch,
        batch_size=128,
        shuffle=True,
        validation_data=(handwritten_validation, synthetic_validation),
        callbacks=[tensorboard_callback])

    predictions = autoencoder.predict(handwritten_test)
    history = autoencoder.evaluate(handwritten_test, synthetic_test, batch_size=128)
    autoencoder.save('modelh5/auto-encoder-model.h5')
    # autoencoder.save(cfg.logs_directory+"auto-encoder-model")
    # testing some random samples
    display(handwritten_test, predictions)

if __name__ == '__main__':
    train_model()
