import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

train_df = pd.read_csv("./data/training_data.csv")
test_df  = pd.read_csv("./data/test_data.csv")

print(train_df)
print(test_df)

X_train = train_df.drop(columns=['age_at_death'])
y_train = train_df['age_at_death']

X_test = test_df.drop(columns=['age_at_death'])
y_test = test_df['age_at_death']

print(X_train.dtypes)
print(y_train.dtypes)

# Build the feedforward neural network (similar to OLS regression)
print("\nBuilding feedforward neural network...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),  # Extra layer for more complex patterns
    tf.keras.layers.Dense(1)  # Output layer for regression (no activation)
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error',
    metrics=['mae']  # Mean Absolute Error as additional metric
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

os.makedirs("learningBase", exist_ok=True)

# metrics on test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

# save metrics
with open("learningBase/training_stats.txt", "w") as f:
    f.write(f"Final train loss: {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final train MAE: {history.history['mae'][-1]:.4f}\n")
    f.write(f"Final val loss: {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Final val MAE: {history.history['val_mae'][-1]:.4f}\n")
    f.write(f"Test loss (MSE): {test_loss:.4f}\n")
    f.write(f"Test MAE: {test_mae:.4f}\n")
    f.write(f"Number of training epochs: {len(history.history["loss"])}\n")


# loss curves
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learningBase/loss_curves.png", dpi=150)
plt.close()

# MAE curves
plt.figure()
plt.plot(history.history["mae"], label="train")
plt.plot(history.history["val_mae"], label="val")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learningBase/mae_curves.png", dpi=150)
plt.close()

# predictions vs truth scatter
y_pred = model.predict(X_test).ravel()
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("True age_at_death")
plt.ylabel("Predicted age_at_death")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", linewidth=2, label='Perfect Prediction')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learningBase/true_vs_pred_scatter.png", dpi=150)
plt.close()

# basic residuals
residuals = y_test - y_pred

# 1) Residuals vs fitted (diagnose non-linearity, heteroscedasticity)
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values (y_pred)")
plt.ylabel("Residuals (y_test - y_pred)")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.savefig("learningBase/residuals_vs_fitted.png", dpi=150)
plt.close()

# 2) Q–Q plot of residuals (diagnose normality)
plt.figure()
sm.qqplot(residuals, line="45", fit=True)
plt.title("Normal Q-Q plot of residuals")
plt.tight_layout()
plt.savefig("learningBase/residuals_qqplot.png", dpi=150)
plt.close()

# 3) Scale–location: sqrt(|residuals|) vs fitted
plt.figure()
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.4)
plt.xlabel("Fitted values (y_pred)")
plt.ylabel("sqrt(|Residuals|)")
plt.title("Scale-Location Plot")
plt.tight_layout()
plt.savefig("learningBase/scale_location.png", dpi=150)
plt.close()

model.save("models/currentAiSolution.h5")

# ============================================================================
# MODEL PERFORMANCE EVALUATION
# ============================================================================

print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

# Get predictions on training and test sets
y_train_pred = model.predict(X_train, verbose=0).flatten()
y_test_pred = model.predict(X_test, verbose=0).flatten()

# Flatten y arrays if needed
y_train_flat = y_train.flatten() if len(y_train.shape) > 1 else y_train
y_test_flat = y_test.flatten() if len(y_test.shape) > 1 else y_test

# Calculate metrics for TRAINING set
train_mse = mean_squared_error(y_train_flat, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train_flat, y_train_pred)
train_r2 = r2_score(y_train_flat, y_train_pred)

print("\n--- TRAINING SET PERFORMANCE ---")
print(f"Mean Squared Error (MSE):       {train_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.4f} years")
print(f"Mean Absolute Error (MAE):      {train_mae:.4f} years")
print(f"R² Score:                       {train_r2:.4f}")
print(f"  → {train_r2*100:.2f}% of variance explained by the model")

# Calculate metrics for TEST set
test_mse = mean_squared_error(y_test_flat, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test_flat, y_test_pred)
test_r2 = r2_score(y_test_flat, y_test_pred)

print("\n--- TEST SET PERFORMANCE (Model Generalization) ---")
print(f"Mean Squared Error (MSE):       {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f} years")
print(f"Mean Absolute Error (MAE):      {test_mae:.4f} years")
print(f"R² Score:                       {test_r2:.4f}")
print(f"  → {test_r2*100:.2f}% of variance explained by the model")

# Interpretation
print("\n--- INTERPRETATION ---")
if test_r2 > 0.8:
    print("✓ Excellent model performance (R² > 0.8)")
elif test_r2 > 0.6:
    print("✓ Good model performance (R² > 0.6)")
elif test_r2 > 0.4:
    print("○ Moderate model performance (R² > 0.4)")
else:
    print("✗ Poor model performance (R² < 0.4)")

print(f"\nOn average, predictions are off by {test_mae:.2f} years (MAE)")
print(f"The model explains {test_r2*100:.2f}% of the variance in age at death")

# Check for overfitting
r2_diff = abs(train_r2 - test_r2)
if r2_diff < 0.05:
    print("✓ No significant overfitting detected")
elif r2_diff < 0.15:
    print("○ Slight overfitting (train R² > test R²)")
else:
    print("✗ Significant overfitting detected")

print("="*70)