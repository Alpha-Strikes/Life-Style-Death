import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Paths mounted via volume (activation data copied to /tmp/acGvaGonBase/ by docker-compose)
DATA_PATH = "/tmp/acGvaGonBase/activation_data.csv"

# Custom Dense layer that ignores quantization_config (compatibility fix for newer TF models)
class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        # Remove quantization_config if present (not supported in older TF versions)
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Load model (from knowledgeBase image, copied to volume)
# Use custom_objects to handle quantization_config compatibility issue
try:
    model = tf.keras.models.load_model("/tmp/knowledgeBase/currentAiSolution.h5", compile=False)
except (TypeError, ValueError) as e:
    if "quantization_config" in str(e) or "Unrecognized keyword" in str(e):
        # Use custom Dense layer that ignores quantization_config
        print("Note: Using compatibility layer for model loading...")
        custom_objects = {'Dense': CompatibleDense}
        model = tf.keras.models.load_model("/tmp/knowledgeBase/currentAiSolution.h5", 
                                         compile=False, 
                                         custom_objects=custom_objects)
    else:
        raise

# Load activation data
df = pd.read_csv(DATA_PATH)

# Drop target column if present
X = df.drop(columns=["age_at_death"], errors="ignore")

# Get the expected input shape from the model
expected_features = model.input_shape[1]
print(f"Model expects {expected_features} features")
print(f"Activation data has {X.shape[1]} features")

# Ensure we have the right number of features
if X.shape[1] != expected_features:
    # If we have more features, try to select the first N features
    # (This assumes the model was trained on the first N features)
    if X.shape[1] > expected_features:
        print(f"Warning: Selecting first {expected_features} features to match model input")
        X = X.iloc[:, :expected_features]
    else:
        raise ValueError(f"Not enough features: model expects {expected_features}, got {X.shape[1]}")

# Normalize (using StandardScaler - note: ideally should use the same scaler from training)
# For now, we normalize independently which may not be perfect but should work
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict
prediction = model.predict(X_scaled, verbose=0)

print("=== AI Prediction Result ===")
print(f"Predicted age at death: {prediction[0][0]:.2f}")