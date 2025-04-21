import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = np.array(p).reshape(self.size, 1)
            self.weights += p @ p.T
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=10):
        pattern = np.array(pattern).copy()
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
            pattern[pattern == 0] = 1
        return pattern

def bin_to_bipolar(vec):
    return np.array([1 if x == 1 else -1 for x in vec])

def bipolar_to_bin(vec):
    return [1 if x >= 0 else 0 for x in vec]

# ✅ Use ONLY TWO patterns to stay under capacity
patterns = [
    bin_to_bipolar([1, 0, 1, 0]),
    bin_to_bipolar([0, 1, 0, 1])
]

net = HopfieldNetwork(size=4)
net.train(patterns)

# Noisy input: missing 3rd bit from [1, 0, 1, 0]
test_input = bin_to_bipolar([1, 0, 0, 0])
recalled = net.recall(test_input)

print("Original input (noisy):", bipolar_to_bin(test_input))
print("Recalled pattern:       ", bipolar_to_bin(recalled))
print("\nWeight matrix:")
print(np.round(net.weights, 2))
