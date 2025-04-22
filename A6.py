"""Implement Artificial Neural Network training process in Python by using Forward Propagation, 
Back Propagation. """

import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (XOR problem as an example)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases randomly
np.random.seed(42)
input_layer_neurons = X.shape[1]   # 2
hidden_layer_neurons = 4
output_neurons = 1

# Weights
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

# Training the ANN
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # FORWARD PROPAGATION
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    predicted_output = sigmoid(final_input)

    # BACK PROPAGATION
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    # Updating Weights and Biases
    wo += hidden_output.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    # Optional: Print error every 1000 iterations
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Error: {np.mean(np.abs(error))}")

# Final Output
print("\nFinal predicted output:")
print(predicted_output)
