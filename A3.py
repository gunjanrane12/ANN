"""Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. 
Given numbers are in ASCII form 0 to 9 """

#Assignment 3

import numpy as np

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Training data: digits 0-9 in 7-bit ASCII binary
X = []
y = []

for i in range(10):
    binary = list(map(int, format(ord(str(i)), '07b')))
    X.append(binary)
    y.append(1 if i % 2 == 1 else 0)  # 1 = odd, 0 = even

X = np.array(X)
y = np.array(y)

# Initialize weights and bias
weights = np.zeros(7)
bias = 0
lr = 0.1  # learning rate

# Training the perceptron
for epoch in range(10):
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        pred = step(z)
        error = y[i] - pred
        weights += lr * error * X[i]
        bias += lr * error

# Test the perceptron
print("Digit Classification (0 = Even, 1 = Odd):")
for i in range(10):
    x_test = np.array(list(map(int, format(ord(str(i)), '07b'))))
    z = np.dot(x_test, weights) + bias
    output = step(z)
    print(f"Digit: {i}, Predicted: {'Odd' if output == 1 else 'Even'}")
