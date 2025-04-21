import numpy as np
import matplotlib.pyplot as plt

# Generate a simple linearly separable dataset
def generate_data():
    np.random.seed(1)
    X1 = np.random.randn(10, 2) + np.array([2, 2])  # Class 1
    X2 = np.random.randn(10, 2) + np.array([-2, -2])  # Class -1
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(10), -np.ones(10)))
    return X, y

# Perceptron Learning Algorithm
def perceptron_train(X, y, epochs=10, lr=0.1):
    w = np.zeros(X.shape[1] + 1)  # Initialize weights
    for _ in range(epochs):
        for i in range(len(y)):
            x_i = np.insert(X[i], 0, 1)  # Insert bias term
            y_pred = np.sign(np.dot(w, x_i))
            if y_pred != y[i]:
                w += lr * y[i] * x_i  # Update rule
    return w

# Function to plot decision boundary
def plot_decision_boundary(X, y, w):
    plt.figure(figsize=(6,6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    
    # Plot decision boundary
    x_vals = np.linspace(-4, 4, 100)
    y_vals = -(w[1] / w[2]) * x_vals - (w[0] / w[2])
    plt.plot(x_vals, y_vals, 'k--', label="Decision Boundary")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Perceptron Decision Boundary")
    plt.show()

# Main execution
X, y = generate_data()
w = perceptron_train(X, y)
plot_decision_boundary(X, y, w)
