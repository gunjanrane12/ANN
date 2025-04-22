"""Write Python program to implement CNN object detection. Discuss numerous performance 
evaluation metrics for evaluating the object detecting algorithms' performance."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random


# Generate synthetic data: images with a white square on black background
def generate_data(num_samples=1000, img_size=64):
    X = []
    y_class = []  # classification label (always 1 here for "object present")
    y_bbox = []  # bounding box [x, y, width, height]

    for _ in range(num_samples):
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        size = random.randint(10, 20)
        x = random.randint(0, img_size - size)
        y = random.randint(0, img_size - size)
        img[y:y + size, x:x + size] = 255

        X.append(img)
        y_class.append(1)  # object exists
        y_bbox.append([x / img_size, y / img_size, size / img_size, size / img_size])  # normalized

    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    y_class = np.array(y_class)
    y_bbox = np.array(y_bbox)
    return X, y_class, y_bbox


# Create training data
X, y_class, y_bbox = generate_data(1000)
X_test, y_class_test, y_bbox_test = generate_data(100)

# Build a simple CNN model
inputs = layers.Input(shape=(64, 64, 1))
x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

# Two heads: one for classification, one for bounding box
output_class = layers.Dense(1, activation='sigmoid', name='class_output')(x)
output_bbox = layers.Dense(4, activation='sigmoid', name='bbox_output')(x)

model = models.Model(inputs=inputs, outputs=[output_class, output_bbox])
model.compile(optimizer='adam',
              loss={'class_output': 'binary_crossentropy', 'bbox_output': 'mse'},
              metrics={'class_output': 'accuracy'})

# Train model
model.fit(X, {'class_output': y_class, 'bbox_output': y_bbox}, epochs=10, batch_size=32)

# Evaluate model
loss, class_loss, bbox_loss, class_acc = model.evaluate(X_test,
                                                        {'class_output': y_class_test, 'bbox_output': y_bbox_test})
print(f"\nClassification Accuracy: {class_acc:.2f}")


# Show predictions
def show_prediction(index):
    img = X_test[index].reshape(64, 64)
    pred_class, pred_bbox = model.predict(X_test[index].reshape(1, 64, 64, 1))
    x, y, w, h = pred_bbox[0] * 64

    plt.imshow(img, cmap='gray')
    if pred_class > 0.5:
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title(f"Predicted Class: {'Object' if pred_class > 0.5 else 'None'}")
    plt.axis('off')
    plt.show()


# Display 3 random test results
for i in random.sample(range(100), 3):
    show_prediction(i)
