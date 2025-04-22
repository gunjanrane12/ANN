"""Write a python program to show Back Propagation Network for XOR function with Binary Input 
and Output """

#Assignment 7

import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)

# XOR input and output (binary)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights and biases
np.random.seed(0)
w1 = np.random.rand(2, 2)
b1 = np.zeros((1, 2))
w2 = np.random.rand(2, 1)
b2 = np.zeros((1, 1))

# Training loop
for epoch in range(10000):
    # Forward pass
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Backward pass
    error = y - a2
    d_output = error * sigmoid_deriv(a2)
    d_hidden = d_output.dot(w2.T) * sigmoid_deriv(a1)

    # Update weights and biases
    w2 += a1.T.dot(d_output) * 0.1
    b2 += np.sum(d_output) * 0.1
    w1 += X.T.dot(d_hidden) * 0.1
    b1 += np.sum(d_hidden) * 0.1
    
    if epoch % 1000 == 0: 
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}") 

# Test output
print("Final output after training:")
print(np.round(a2, 3))
