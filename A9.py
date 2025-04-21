import numpy as np

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset (XOR logic gate)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Set random seed for reproducibility
np.random.seed(42)

# Neural Network Architecture
input_layer_neurons = X.shape[1]    # 2 input neurons
hidden_layer_neurons = 2            # You can adjust this
output_neurons = 1                  # 1 output neuron

# Initialize weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # -------- FORWARD PROPAGATION --------
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # -------- BACKPROPAGATION --------
    # Error in output
    error = y - predicted_output

    # Gradient for output layer
    d_output = error * sigmoid_derivative(predicted_output)

    # Error for hidden layer
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # -------- WEIGHT AND BIAS UPDATES --------
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Optional: Print error every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# -------- FINAL OUTPUT --------
print("\nFinal predictions after training:")
print(predicted_output.round(3))
