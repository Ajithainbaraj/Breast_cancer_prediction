# dnn_model.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
import joblib

# ----------------------------
# Load and Prepare Dataset
# ----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# ----------------------------
# Select Important Features
# ----------------------------
important_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness"
]

important_idx = [list(feature_names).index(f) for f in important_features]
X_selected = X[:, important_idx]

# ----------------------------
# Scale Features
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ----------------------------
# Split Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------------
# Build Deep Neural Network
# ----------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ----------------------------
# Train the Model
# ----------------------------
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# ----------------------------
# Evaluate and Save
# ----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ DNN Model Accuracy: {acc:.4f}")

# Save the model
model.save("breast_cancer_dnn.h5")
print("✅ DNN Model Saved Successfully!")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler Saved Successfully!")

# Save the important feature names
np.save("important_features.npy", important_features)
print("✅ Important Features Saved Successfully!")
