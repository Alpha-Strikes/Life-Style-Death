import keras

keras.models.load_model(
    "models/currentAiSolution_INFERENCE.keras",
    compile=False
)
print("✅ Clean model loads without errors")