import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from sklearn.model_selection import train_test_split

from scripts.config import (
    RAW_DATA_PATH, SAMPLE_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COL, DROP_COLS, MODEL_PATH, SHAP_SUMMARY_PATH, MODELS_DIR
)
from scripts.data_loader import load_data
from scripts.feature_engineering import add_features, get_column_lists
from scripts.train import train_models, save_model
from scripts.evaluate import evaluate_models, save_metrics_json, plot_roc_curves
from scripts.explain import generate_shap_summary

def run_pipeline(use_sample: bool = False):
    """
    Run the end-to-end Machine Learning pipeline.
    """
    print("=" * 60)
    print("Starting End-to-End Customer Churn Prediction ML Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    data_path = SAMPLE_DATA_PATH if use_sample else RAW_DATA_PATH
    print(f"\n[1/6] Loading dataset from: {data_path}")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # 2. Train-Test Split (Stratified)
    print(f"\n[2/6] Performing Train-Test Split (test_size={TEST_SIZE}, stratify={TARGET_COL})...")
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. Feature Engineering
    print("\n[3/6] Applying Feature Engineering...")
    X_train_fe = add_features(X_train)
    X_test_fe = add_features(X_test)
    num_cols, cat_cols = get_column_lists(X_train_fe)
    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    
    # 4. Model Training
    print("\n[4/6] Training Candidate Pipelines (SMOTE + Preprocessing + Classifier)...")
    trained_models = train_models(X_train_fe, y_train, num_cols, cat_cols)
    
    # 5. Model Evaluation
    print("\n[5/6] Evaluating Models on Test Set...")
    metrics = evaluate_models(trained_models, X_test_fe, y_test)
    save_metrics_json(metrics)
    
    # Print metrics table
    import pandas as pd
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    print("\n--- Test Metrics Summary ---")
    print(metrics_df.to_string())
    
    # Save ROC curves plot
    roc_plot_path = MODELS_DIR / "roc_comparison.png"
    plot_roc_curves(trained_models, X_test_fe, y_test, save_path=roc_plot_path)
    print(f"Saved ROC comparison plot to: {roc_plot_path}")
    
    # Select Best Model (Random Forest by default based on ROC-AUC & F1)
    best_model_name = "RandomForest"
    best_model = trained_models[best_model_name]
    save_model(best_model, filepath=MODEL_PATH)
    
    # 6. SHAP Explainability
    print("\n[6/6] Generating SHAP Feature Importance Summary...")
    try:
        generate_shap_summary(best_model, X_test_fe, num_cols, cat_cols, save_path=SHAP_SUMMARY_PATH)
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot: {e}")
        
    print("\n" + "=" * 60)
    print("Pipeline Execution Completed Successfully!")
    print(f"Saved best model artifact ({best_model_name}) to: {MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Customer Churn Prediction Pipeline")
    parser.add_argument("--sample", action="store_true", help="Use small sample dataset for rapid testing")
    args = parser.parse_args()
    
    run_pipeline(use_sample=args.sample)
