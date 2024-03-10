import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import mnist
from config import Config as cfg

def processing(image_list):
    """
    Preprocesses a NumPy array representing images.

    Parameters:
    - array (numpy.ndarray): An array of image data.

    Returns:
    - numpy.ndarray: The preprocessed array.

    The function performs the following preprocessing steps:
    1. Converts the array to 'float32' type.
    2. Normalizes pixel values to the range [0, 1] by dividing by 255.0.
    3. Reshapes the array to have dimensions (batch_size, 28, 28, 1), assuming each image is 28x28 pixels.

    # Returns a processed array with normalized pixel values and the correct shape.
    """
    array = image_list.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array


def synthetic_data_adder(array):
    """
    Load and preprocess synthetic images corresponding to given labels.

    Parameters:
    - array (numpy.ndarray): Array of label values.

    Returns:
    - numpy.ndarray: Processed images.

    Reads and normalizes grayscale images based on provided labels.
    Reshapes images to (batch_size, 28, 28, 1).
    """
    image_list = []
    for i in range(0, len(array)):
        img = cv2.imread(cfg.image_directory + str(array[i]) + ".jpg", cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        img = img / 255.0
        image_list.append(img)
    data = np.reshape(image_list, (len(array), 28, 28, 1))
    return data

def data_splitting(data):
    """
    Split the input data into train data and validation sets as there is no validation data for mnist.

    Parameters:
    - data: Input data with shape (num_samples, height, width, channels)

    Returns:
    - train_data: First 90% of data
    - validation_data: Remaining 10% of data
    """
    num_samples = data.shape[0]

    # 10% data is stored for validation from the train dataset
    split_index = num_samples - int((num_samples*10)/100)

    # Split the dataset
    train_data = data[:split_index]
    validation_data = data[split_index:]

    return train_data, validation_data
def load_dataset():
    """
    Load and preprocess the MNIST dataset for letters.

    Returns:
    - Tuple of numpy.ndarrays: Preprocessed data for handwritten and synthetic letters.
      Format: (handwritten_train_data, synthetic_train_data,
               handwritten_validation_data, validation_synthetic_data,
               handwritten_test_data, test_synthetic_data)
    """
    (raw_train_data, train_labels), (raw_test_data, test_labels) = mnist.load_data()

    # Normalize and reshape the data
    handwritten_train_data = processing(raw_train_data)
    handwritten_test_data = processing(raw_test_data)

    synthetic_train_data = synthetic_data_adder(train_labels)
    synthetic_test_data = synthetic_data_adder(test_labels)

    handwritten_train_data, handwritten_validation_data = data_splitting(handwritten_train_data)
    synthetic_train_data, validation_synthetic_data = data_splitting(synthetic_train_data)
    return handwritten_train_data, synthetic_train_data, handwritten_validation_data, validation_synthetic_data, handwritten_test_data, synthetic_test_data

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    n = 15
    # print("-------",array1.shape)
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    # print("-------",images1.shape)
    images2 = array2[indices, :]
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
