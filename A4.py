"""With a suitable example demonstrate the perceptron learning law with its decision regions using 
python. Give the output in graphical form. """

import numpy as np
import matplotlib.pyplot as plt

# Input data (AND gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
lr = 0.1  # learning rate

# Perceptron learning rule
for epoch in range(10):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1 if z >= 0 else 0
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

# Plotting decision boundary
x1 = np.linspace(-0.2, 1.2, 100)
x2 = -(w[0]*x1 + b) / w[1]

plt.figure(figsize=(8,6))
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], color='red' if y[i] == 1 else 'blue')

plt.plot(x1, x2, label='Decision Boundary', color='green')
plt.title('Perceptron Decision Region (AND Gate)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.legend()
plt.show()
