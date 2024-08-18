import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Reshape the data to include the channel dimension (28, 28, 1)
# This is done since the convolutional neural network (CNN) requires 3 dimensions where the last dimension is the
# number of channels (colors). It's 1 because it is greyscale.
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # first number is the no. of filters
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # converts 2D to 1D for the dense layer
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# No need one-hot encoding when using sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

model.save('third-new-model.keras')

model = models.load_model('third-new-model.keras')

# Function to load and preprocess image using OpenCV
def PreprocessImage(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
    
    # Get the target size and the current aspect ratio
    target_size = (28, 28)
    height, width = img.shape
    aspect_ratio = width / height
    
    # Compute padding to maintain aspect ratio
    if aspect_ratio > 1:
        new_width = target_size[1]
        new_height = int(new_width / aspect_ratio)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pad_vert = (target_size[0] - new_height) // 2
        padded_img = cv2.copyMakeBorder(resized_img, pad_vert, target_size[0] - new_height - pad_vert, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        new_height = target_size[0]
        new_width = int(new_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pad_horz = (target_size[1] - new_width) // 2
        padded_img = cv2.copyMakeBorder(resized_img, 0, 0, pad_horz, target_size[1] - new_width - pad_horz, cv2.BORDER_CONSTANT, value=0)
    
    # Invert colors, normalize and add channel and batch dimensions
    padded_img = np.invert(padded_img)
    padded_img = tf.keras.utils.normalize(padded_img, axis=1)
    padded_img = np.expand_dims(padded_img, axis=-1)
    padded_img = np.expand_dims(padded_img, axis=0)
    
    return padded_img

def PredictDigit():
    image_number = 1
    while os.path.isfile(f"digits2/digit{image_number}.png"):
        try:
            img = PreprocessImage(f"digits2/digit{image_number}.png")
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")
        finally:
            image_number += 1


PredictDigit()