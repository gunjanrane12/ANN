"""Write a python program to illustrate ART neural network."""

#Assignment 8

import numpy as np

# Parameters
vigilance = 0.8

# Input patterns (binary)
patterns = [
    np.array([1, 0, 0, 1]),
    np.array([1, 1, 0, 1]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 1, 1])
]

# Store categories
categories = []

def match_category(input_pattern, category):
    # Check match score (overlap)
    match = np.sum(np.logical_and(input_pattern, category)) / np.sum(input_pattern)
    return match >= vigilance

def train_art(patterns):
    for p in patterns:
        matched = False
        for i, cat in enumerate(categories):
            if match_category(p, cat):
                # Update existing category (AND rule)
                categories[i] = np.logical_and(categories[i], p)
                matched = True
                break
        if not matched:
            categories.append(p.copy())

# Train ART network
train_art(patterns)

# Output the result
print("Formed categories:")
for i, c in enumerate(categories):
    print(f"Category {i+1}: {c.astype(int)}")
