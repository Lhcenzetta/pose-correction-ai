import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skreprocessinglearn.p import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("keypoints_dataset.csv")
X = df.drop("label", axis=1).values
y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

model = Sequential(
    [
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

early_stop = EarlyStopping(patience=20)

history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy:{acc*100:.1f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassificatioortn Rep:")
print(classification_report(y_test, y_pred, target_names=["Incorrect", "Correct"]))

cm = confusion_matrix(y_test, y_pred)
print(cm)

model.save("shoulder_model.keras")

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
