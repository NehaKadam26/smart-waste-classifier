"""
Per-class evaluation for Smart Waste Classifier.
Run this from the project root (same level as model/ and data/).

Outputs:
  - results/classification_report.csv   (per-class precision/recall/F1/support)
  - results/confusion_matrix.png        (visual confusion matrix)
  - results/per_class_summary.md        (markdown table, paste straight into README)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "model/model.h5"
CLASS_INDEX_PATH = "model/class_indices.json"
VAL_DIR = "data/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUT_DIR = "results"

os.makedirs(OUT_DIR, exist_ok=True)

# --- Load model + class mapping ---
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)  # e.g. {"battery": 0, "biological": 1, ...}

# invert to index -> name, ordered by index
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# --- Load validation set (no shuffle, so predictions line up with labels) ---
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

# Keras infers class order from folder names alphabetically — capture it BEFORE
# .map() below, since .map() returns a plain MapDataset that loses this attribute.
inferred_classes = val_ds.class_names

# IMPORTANT: MobileNetV2 expects inputs scaled to [-1, 1], not raw [0, 255] pixels.
# image_dataset_from_directory loads raw pixel values, so we must apply the same
# preprocessing used at training time or predictions will be near-random.
# If your app.py uses a different scaling (e.g. plain /255.0), change this line to match.
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

# confirm folder order matches class_indices.json
assert inferred_classes == class_names, (
    f"Class order mismatch!\nFolder order: {inferred_classes}\nclass_indices.json order: {class_names}\n"
    "Fix by re-checking class_indices.json before trusting results."
)

# --- Predict ---
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Classification report (per-class precision/recall/F1/support) ---
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(f"{OUT_DIR}/classification_report.csv")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# --- Confusion matrix plot ---
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
print(f"\nSaved confusion matrix to {OUT_DIR}/confusion_matrix.png")

# --- Markdown table for README, sorted worst-to-best by F1 ---
per_class = report_df.iloc[:len(class_names)][["precision", "recall", "f1-score", "support"]]
per_class = per_class.sort_values("f1-score")

md_lines = ["| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
for cls, row in per_class.iterrows():
    md_lines.append(
        f"| {cls} | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1-score']:.2f} | {int(row['support'])} |"
    )

with open(f"{OUT_DIR}/per_class_summary.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"\nSaved README-ready table to {OUT_DIR}/per_class_summary.md")
print("\nWeakest classes (lowest F1) — these are your 'before' numbers if you retrain with fixes:")
print(per_class.head(3))
