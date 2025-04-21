import numpy as np

class ART1:
    def __init__(self, input_size, num_categories, vigilance=0.8):
        self.input_size = input_size
        self.num_categories = num_categories
        self.vigilance = vigilance
        self.wf = np.ones((num_categories, input_size))  # Feedforward weights
        self.wb = np.ones((num_categories, input_size))  # Feedback weights
        self.categories_learned = [False] * num_categories

    def train(self, inputs):
        for input_vector in inputs:
            input_vector = np.array(input_vector)
            print(f"\nInput: {input_vector}")
            matched = False

            for j in range(self.num_categories):
                if not self.categories_learned[j]:
                    continue

                # Calculate match score (dot product)
                yj = np.sum(np.minimum(input_vector, self.wf[j]))

                norm_input = np.sum(input_vector)
                if norm_input == 0:
                    similarity = 0
                else:
                    similarity = yj / norm_input

                if similarity >= self.vigilance:
                    print(f"Matched category {j} with similarity {similarity:.2f}")
                    self.update_weights(j, input_vector)
                    matched = True
                    break
                else:
                    print(f"Category {j} failed vigilance test: {similarity:.2f}")

            if not matched:
                # Find an unused category
                for j in range(self.num_categories):
                    if not self.categories_learned[j]:
                        print(f"No match found. Learning new category {j}.")
                        self.wf[j] = input_vector.copy()
                        self.wb[j] = input_vector.copy()
                        self.categories_learned[j] = True
                        break

    def update_weights(self, category, input_vector):
        # Update weights using intersection (logical AND)
        self.wf[category] = np.minimum(self.wf[category], input_vector)
        self.wb[category] = np.minimum(self.wb[category], input_vector)

    def predict(self, input_vector):
        input_vector = np.array(input_vector)
        for j in range(self.num_categories):
            if not self.categories_learned[j]:
                continue
            yj = np.sum(np.minimum(input_vector, self.wf[j]))
            norm_input = np.sum(input_vector)
            if norm_input == 0:
                similarity = 0
            else:
                similarity = yj / norm_input

            if similarity >= self.vigilance:
                return j
        return -1  # No match found

# Example usage
if __name__ == "__main__":
    data = [
        [1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
    ]

    art = ART1(input_size=5, num_categories=5, vigilance=0.75)
    art.train(data)

    # Test predictions
    test = [0, 1, 1, 0, 1]
    predicted_cat = art.predict(test)
    print(f"\nTest pattern {test} predicted category: {predicted_cat}")

