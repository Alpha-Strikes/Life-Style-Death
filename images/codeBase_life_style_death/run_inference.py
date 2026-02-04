import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths mounted via volume
DATA_PATH = "/tmp/activationBase/activation_data.csv"

# Load model
model = tf.keras.models.load_model( "/tmp/knowledgeBase/currentAiSolution_INFERENCE.keras", compile=False)

# Load activation data
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["age_at_death"], errors="ignore")

# Normalize (simple runtime scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict
prediction = model.predict(X_scaled)

print("=== AI Prediction Result ===")
print(f"Predicted age at death: {prediction[0][0]:.2f}")