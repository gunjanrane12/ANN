"""Write a python Program for Bidirectional Associative Memory with two pairs of vectors. """

#Assignment 5

import numpy as np

# Define bipolar input-output pairs
X = np.array([[1, -1, 1], [-1, 1, -1]])   # 2 input vectors (3 elements)
Y = np.array([[1, 1], [-1, -1]])          # 2 output vectors (2 elements)

# Calculate weight matrix (Hebbian learning)
W = X.T @ Y  # W = ∑{  X[i](transpose) × Y  }

# Test BAM recall
def recall_bam(x_input):
    y = np.sign(x_input @ W)
    x_recalled = np.sign(y @ W.T)
    return y, x_recalled

# Try recalling both patterns
for i in range(len(X)):
    print(f"\nInput X: {X[i]}")
    y_out, x_recalled = recall_bam(X[i])
    print(f"Recalled Y: {y_out}")
    print(f"Recalled X: {x_recalled}")
