"""How to Train a Neural Network with TensorFlow/Pytorch and evaluation of logistic regression
using tensorflow """

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# 1️ Data Preprocessing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data to range [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten the images to 1D arrays (28 * 28 = 784)
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# 2️ Logistic Regression Model (as a simple neural network with one layer)
model_lr = models.Sequential([
    layers.InputLayer(input_shape=(28*28,)),  # Input layer for 784 features
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes for digits 0-9
])

# 3️ Neural Network Model (with hidden layers)
model_nn = models.Sequential([
    layers.InputLayer(input_shape=(28*28,)),  # Input layer for 784 features
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    layers.Dense(64, activation='relu'),   # Another hidden layer with 64 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 classes for digits 0-9
])

# 4️ Compile Logistic Regression Model
model_lr.compile(optimizer='sgd',  # SGD (Stochastic Gradient Descent) optimizer for logistic regression
                 loss='sparse_categorical_crossentropy',  # Sparse because labels are integers
                 metrics=['accuracy'])

# 5️ Compile Neural Network Model
model_nn.compile(optimizer='adam',  # Adam optimizer is widely used for neural networks
                 loss='sparse_categorical_crossentropy',  # Sparse categorical cross-entropy for integer labels
                 metrics=['accuracy'])

# 6️ Train Logistic Regression Model
print("Training Logistic Regression Model...")
model_lr.fit(X_train, y_train, epochs=5, batch_size=64)

# 7️ Train Neural Network Model
print("\nTraining Neural Network Model...")
model_nn.fit(X_train, y_train, epochs=5, batch_size=64)

# 8️ Evaluate Logistic Regression Model on Test Data
print("\nEvaluating Logistic Regression Model on Test Data...")
test_loss_lr, test_acc_lr = model_lr.evaluate(X_test, y_test)
print(f"Test Accuracy of Logistic Regression: {test_acc_lr * 100:.2f}%")

# 9️ Evaluate Neural Network Model on Test Data
print("\nEvaluating Neural Network Model on Test Data...")
test_loss_nn, test_acc_nn = model_nn.evaluate(X_test, y_test)
print(f"Test Accuracy of Neural Network: {test_acc_nn * 100:.2f}%")
