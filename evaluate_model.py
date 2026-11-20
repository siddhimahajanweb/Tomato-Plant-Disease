
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE_DIR = r"D:\Tomato Plant Disease"
MODEL_PATH = os.path.join(BASE_DIR, "tomato_care_cnn.keras")
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "test")

class_names = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria_leaf_spot",
    "target_spot",
    "twospotted_spider_mite",
    "yellow_leaf_curl_virus"
]

present_classes = [
    name for name in class_names
    if os.path.isdir(os.path.join(TEST_DIR, name))
]

print("Test classes found:", present_classes)
print("Missing test classes:", [
    name for name in class_names if name not in present_classes
])

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nLoading test images...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=present_classes,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

test_dataset = test_dataset.map(
    lambda images, labels: (preprocess_input(images), labels)
)

y_true_test = np.concatenate(
    [labels.numpy() for _, labels in test_dataset]
)

print("\nPredicting...")
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Convert test labels to the model's original 10-class order
present_indices = [class_names.index(name) for name in present_classes]
y_true = np.array([present_indices[i] for i in y_true_test])

print("\nTEST ACCURACY:", round(accuracy_score(y_true, y_pred), 4))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

output_path = os.path.join(BASE_DIR, "confusion_matrix.csv")
cm_df.to_csv(output_path)

print("\nConfusion matrix saved to:", output_path)
print("Evaluation completed.")
