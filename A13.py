"MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow"

import tensorflow as tf
import os


#using this we can directly load data into minst variable
minst=tf.keras.datasets.mnist

(X, Y), (x,y) = minst.load_data()

X=X/255.0
x=x/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
                                    ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x,y,epochs=15)
validation_data=(x,y)

test_loss, test_acc = model.evaluate(x, y)
print(f'Test accuracy: {test_acc}')
model.save('Models/model_3_layer.h5')


#PREDICTOR
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("E:\Github\Digit-Recognition\Models\model_2_layer.h5")


# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image using PIL
    img = Image.open(image_path)

    # Resize the image to 28x28 (MNIST size)
    img_resized = img.resize((28, 28))

    # Normalize the pixel values to the range [0, 1]
    img_array = np.array(img_resized) / 255.0

    # Reshape the image to match the model's input shape (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# Function to predict the digit in the image
def predict_digit(image_path):
    img_array = preprocess_image(image_path)

    # Get the model's prediction
    prediction = model.predict(img_array)

    # Get the predicted digit (the class with the highest probability)
    predicted_digit = np.argmax(prediction, axis=1)[0]

    return predicted_digit


# Path to the image you want to predict
image_path = "E:\\Github\\Digit-Recognition\\Scripts\\6.jpg"  # Change this to your image path

# Make prediction
predicted_digit = predict_digit(image_path)
print(f'Predicted digit: {predicted_digit}')

