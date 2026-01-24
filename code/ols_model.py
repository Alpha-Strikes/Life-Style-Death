import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_URL

from data_prep import load_raw_data, train_test_split_encoded


def train_ols_model(data_url: str = DATASET_URL, test_size: float = 0.2, random_state: int = 42):
    print("=" * 60)
    print("OLS REGRESSION MODEL")
    print("=" * 60)
    
    print("\n1. Loading and preparing data...")
    df = load_raw_data(data_url)
    X_train, X_test, y_train, y_test, feature_names = train_test_split_encoded(
        df, test_size=test_size, random_state=random_state
    )
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test set: {X_test.shape[0]} samples")
    

    
    print("\n2. Fitting OLS model...")
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    
    #for statsmodels OLS, we need to add an intercept term
    X_train_with_intercept = sm.add_constant(X_train, has_constant='add')
    X_test_with_intercept = sm.add_constant(X_test, has_constant='add')
    
    #fit OLS model
    model = sm.OLS(y_train, X_train_with_intercept).fit()
    
    #get predictions
    y_train_pred = model.predict(X_train_with_intercept)
    y_test_pred = model.predict(X_test_with_intercept)
    
    #calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
    }
    
    print("\n3. Model Summary:")
    print(model.summary())
    summary_text = str(model.summary())
    
    print("\n4. Performance Metrics:")
    print(f"   Training Set:")
    print(f"      MAE:  {train_mae:.2f} years")
    print(f"      RMSE: {train_rmse:.2f} years")
    print(f"      R²:   {train_r2:.4f}")
    print(f"   Test Set:")
    print(f"      MAE:  {test_mae:.2f} years")
    print(f"      RMSE: {test_rmse:.2f} years")
    print(f"      R²:   {test_r2:.4f}")
    
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "feature_names": feature_names,
        "metrics": metrics,
        "summary_text": summary_text,
    }


def plot_ols_diagnostics(results: dict, output_dir: str = "figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model = results["model"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    y_train_pred = results["y_train_pred"]
    y_test_pred = results["y_test_pred"]
    
    #get residuals
    residuals_train = y_train - y_train_pred
    residuals_test = y_test - y_test_pred
    
    #get standardized residuals and leverage (for additional diagnostic plots)
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    
    print("\n5. Generating diagnostic plots...")
    
    # 1. Residuals vs Fitted (Training)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_pred, residuals_train, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Fitted Values (Predicted Age at Death)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values (Training Set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_residuals_vs_fitted.png"), dpi=150)
    plt.close()
    
    # 2. Q-Q Plot of Residuals (Training)
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals_train, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals (Training Set)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_qq_plot.png"), dpi=150)
    plt.close()
    
    # 3. Sqrt (Standardized Residual) vs Fitted Values (Scale-Location Plot)
    sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_pred, sqrt_std_residuals, alpha=0.5, s=20)

    #add LOESS-smoothed line using statsmodels' lowess
    # (smooths sqrt(|standardized residuals|) over fitted values)
    try:
        lowess_result = sm.nonparametric.lowess(
            sqrt_std_residuals,
            y_train_pred,
            frac=0.3,#smoothing parameter (0.2–0.4 is typical)
        )
        plt.plot(
            lowess_result[:, 0],
            lowess_result[:, 1],
            "r-",
            linewidth=2,
        )
    except Exception:
        pass

    num_labels = 5
    if len(sqrt_std_residuals) > 0:
        top_indices = np.argsort(sqrt_std_residuals)[-num_labels:]
        for idx in top_indices:
            plt.annotate(
                str(idx),
                (y_train_pred[idx], sqrt_std_residuals[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="red",
                fontweight="bold",
            )

    plt.xlabel("Fitted values")
    plt.ylabel("√|Standardized Residuals|")
    plt.title("Scale-Location")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_scale_location_plot.png"), dpi=150)
    plt.close()
    
    # 4. Residual vs Leverage
    plt.figure(figsize=(8, 6))
    plt.scatter(leverage, residuals_train, alpha=0.5)
    plt.xlabel("Leverage")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Leverage (Training Set)")
    
    #add Cook's distance contours(common thresholds: 0.5& 1.0)
    #Cook's distance= (standardized_residuals^2 *leverage)/(no. of parameters*(1- leverage))
    n_params = len(model.params)
    cook_threshold = 0.5
    leverage_range = np.linspace(leverage.min(), leverage.max(), 100)
    cook_contour_05 = np.sqrt(cook_threshold * n_params * (1 - leverage_range) / leverage_range)
    cook_contour_10 = np.sqrt(1.0 * n_params * (1 - leverage_range) / leverage_range)
    
    plt.plot(leverage_range, cook_contour_05, 'r--', alpha=0.5, label=f"Cook's D = {cook_threshold}")
    plt.plot(leverage_range, -cook_contour_05, 'r--', alpha=0.5)
    plt.plot(leverage_range, cook_contour_10, 'r:', alpha=0.5, label="Cook's D = 1.0")
    plt.plot(leverage_range, -cook_contour_10, 'r:', alpha=0.5)
    
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_residuals_vs_leverage.png"), dpi=150)
    plt.close()

    # 5. Predicted vs Actual (Training)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Age at Death")
    plt.ylabel("Predicted Age at Death")
    plt.title("Predicted vs Actual (Training Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_predicted_vs_actual_train.png"), dpi=150)
    plt.close()
    
    # 6. Predicted vs Actual (Test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Age at Death")
    plt.ylabel("Predicted Age at Death")
    plt.title("Predicted vs Actual (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ols_predicted_vs_actual_test.png"), dpi=150)
    plt.close()
    
    print(f"   Saved plots to {output_dir}/")
    print("      - ols_residuals_vs_fitted.png")
    print("      - ols_qq_plot.png")
    print("      - ols_scale_location_plot.png")
    print("      - ols_residuals_vs_leverage.png")
    print("      - ols_predicted_vs_actual_train.png")
    print("      - ols_predicted_vs_actual_test.png")


def save_ols_results(results: dict, output_dir: str = "outputs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #save full summary
    with open(os.path.join(output_dir, "ols_summary.txt"), "w") as f:
        f.write("OLS REGRESSION MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(results["summary_text"])
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        metrics = results["metrics"]
        f.write(f"Training Set:\n")
        f.write(f"  MAE:  {metrics['train_mae']:.2f} years\n")
        f.write(f"  RMSE: {metrics['train_rmse']:.2f} years\n")
        f.write(f"  R²:   {metrics['train_r2']:.4f}\n\n")
        f.write(f"Test Set:\n")
        f.write(f"  MAE:  {metrics['test_mae']:.2f} years\n")
        f.write(f"  RMSE: {metrics['test_rmse']:.2f} years\n")
        f.write(f"  R²:   {metrics['test_r2']:.4f}\n")
    
    print(f"\n6. Saved results to {output_dir}/ols_summary.txt")


def save_ols_model(results: dict, output_dir: str = "models", learning_base_dir: str = "learningBase"):
    #create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(learning_base_dir, exist_ok=True)
    
    model_data = {
        "model": results["model"],
        "feature_names": results["feature_names"],
        "metrics": results["metrics"],
        "summary_text": results["summary_text"],
    }
    
    #save model as currentOlsSolution.pkl (PDF requirement: currentOlsSolution.xml or appropriate format)
    model_path = os.path.join(output_dir, "currentOlsSolution.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"7. Saved model to {output_dir}/currentOlsSolution.pkl")
    
    #save performance metrics to learningBase
    metrics = results["metrics"]
    metrics_file = os.path.join(learning_base_dir, "ols_performance_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("OLS MODEL PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Training Set:\n")
        f.write(f"  MAE:  {metrics['train_mae']:.4f} years\n")
        f.write(f"  RMSE: {metrics['train_rmse']:.4f} years\n")
        f.write(f"  R²:   {metrics['train_r2']:.4f}\n\n")
        f.write("Test/Validation Set:\n")
        f.write(f"  MAE:  {metrics['test_mae']:.4f} years\n")
        f.write(f"  RMSE: {metrics['test_rmse']:.4f} years\n")
        f.write(f"  R²:   {metrics['test_r2']:.4f}\n")
    
    print(f"   Saved performance metrics to {learning_base_dir}/ols_performance_metrics.txt")


def load_ols_model(model_path: str = "models/ols_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    print(f"Loaded OLS model from {model_path}")
    return model_data


def predict_with_ols_model(model_data: dict, X: np.ndarray):
    #for statsmodels OLS, we need to add an intercept term
    X_with_intercept = sm.add_constant(X, has_constant='add')
    
    #make predictions
    predictions = model_data["model"].predict(X_with_intercept)
    
    return predictions


def run_ols_analysis(data_url: str = DATASET_URL):
    #train model
    results = train_ols_model(data_url)
    
    #generate diagnostic plots
    plot_ols_diagnostics(results)
    
     #save results
    save_ols_results(results)
    
    #save model
    save_ols_model(results)
    
    print("\n" + "=" * 60)
    print("OLS ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_ols_analysis()
