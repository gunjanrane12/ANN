import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_vectors, output_vectors):
        for x, y in zip(input_vectors, output_vectors):
            self.weights += np.outer(x, y)

    def recall_output(self, input_vector):
        return np.sign(np.dot(input_vector, self.weights))

    def recall_input(self, output_vector):
        return np.sign(np.dot(output_vector, self.weights.T))


input_vectors = np.array([[1, 1], [-1, 1]])
output_vectors = np.array([[1, -1], [1, 1]])

bam = BAM(input_size=2, output_size=2)

bam.train(input_vectors, output_vectors)

print("Recall output for input [1, 1]:", bam.recall_output([1, 1]))  
print("Recall output for input [-1, 1]:", bam.recall_output([-1, 1]))  

print("Recall input for output [1, -1]:", bam.recall_input([1, -1]))  
print("Recall input for output [1, 1]:", bam.recall_input([1, 1]))    
