"""
STEP 2 — Train Model for Shoulder Abduction
============================================
Trains an MLP on the 28 features extracted in Step 1.
Also finds the best angle threshold for rule-based fallback.

Output:
    shoulder_model.keras
    scaler.pkl

Install:
    pip install tensorflow scikit-learn pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ── Load data ─────────────────────────────────────────────────────
df = pd.read_csv("keypoints_dataset.csv")
X  = df.drop("label", axis=1).values   # 28 features
y  = df["label"].values                 # 0=incorrect, 1=correct

print(f"Dataset: {len(df)} samples")
print(f"  Correct  : {int(y.sum())}")
print(f"  Incorrect: {int((1-y).sum())}")

# ── Show angle distribution (useful for your report!) ─────────────
print(f"\n📐 Angle summary:")
angle_cols = ["left_abduction_angle", "right_abduction_angle"]
for col in angle_cols:
    if col in df.columns:
        print(f"  {col}:")
        print(f"    Correct   avg: {df[df['label']==1][col].mean():.1f}°")
        print(f"    Incorrect avg: {df[df['label']==0][col].mean():.1f}°")

# ── Train/test split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Normalize ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("\n✅ Scaler saved: scaler.pkl")

# ── Build model ───────────────────────────────────────────────────
# Smaller model since we only have 28 features (not 99)
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Train ─────────────────────────────────────────────────────────
early_stop = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluate ──────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.1f}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Incorrect", "Correct"]))

# ── Confusion matrix ──────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Incorrect", "Correct"])
disp.plot(colorbar=False)
plt.title("Confusion Matrix — Shoulder Abduction Model")
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved: confusion_matrix.png")

# ── Save model ────────────────────────────────────────────────────
model.save("shoulder_model.keras")
print("✅ Model saved: shoulder_model.keras")

# ── Training curves ───────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy — Shoulder Abduction")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print("✅ Training plot saved: training_history.png")