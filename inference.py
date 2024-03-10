import tensorflow as tf 
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_directory = './test_image/'

# Load and preprocess the original image
original_img = cv2.imread(image_directory + "test_image4.png", cv2.IMREAD_GRAYSCALE)
original_img = cv2.resize(original_img, (28, 28)) / 255.
original_img = np.reshape(original_img, (1, 28, 28, 1))

# Load the model
model = tf.keras.models.load_model('modelh5/auto-encoder-model.h5')

# Predict using the model
predictions = model.predict(original_img)

# Plot the original and predicted images side by side
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Plot the original image on the left
axs[0].imshow(np.squeeze(original_img), cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original')

# Plot the predicted image on the right
axs[1].imshow(np.squeeze(predictions), cmap='gray')
axs[1].axis('off')
axs[1].set_title('Generated')
plt.savefig('./test_image/test_image4.png')
plt.show()
