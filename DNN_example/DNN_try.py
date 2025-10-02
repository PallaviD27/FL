'''Steps:
1. Import Libraries
2. Load and pre-process dataset
3. Build the DNN
4. Compile the model
5. Train (fit) the model
6. Evaluate and predict
7. Visualize performance

'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Load data and split it into training and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise images: Scale pixel value [0,255] to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1,784)

# One-hot encoding of labels
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

# Initialize model
model = Sequential()

# Input layer and first hidden layer
model.add(Input(shape=(784,)))
model.add(Dense(units=128, activation='relu'))  # Dense implies fully connected layers - each neuron in the previous layer is connected to each neuron in the next.

# Second hidden layer
model.add(Dense(units=64, activation='relu'))

# Output layer
model.add(Dense(units=10, activation='softmax'))

# Compiling model
model.compile (
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model

model.fit(x_train, y_train, epochs = 10, batch_size=32, validation_split=0.1)


# Load and identify new image
img = image.load_img(r'C:\Users\palla\Downloads\ident3.png', color_mode='grayscale', target_size=(28,28))

# Convert image to array and normalize
img_array = image.img_to_array(img)
img_array =  255 - img_array  # Images to be identified are black ink with white background while MNIST is white on black background
img_array = img_array.reshape(1,784)
img_array = img_array.astype('float32') / 255.0

# Predict the digit
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)

print(f'Predicted digit: {predicted_digit}')


plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis('off')
plt.show()
