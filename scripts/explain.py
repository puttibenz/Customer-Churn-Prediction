import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scripts.config import SHAP_SUMMARY_PATH

def get_transformed_feature_names(preprocessor, num_cols, cat_cols) -> list:
    """
    Extract feature names after ColumnTransformer preprocessing.
    """
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_encoded_names = list(cat_encoder.get_feature_names_out(cat_cols))
    except Exception:
        cat_encoded_names = list(cat_cols)
        
    return list(num_cols) + cat_encoded_names

def generate_shap_summary(pipeline, X_test: pd.DataFrame, num_cols: list, cat_cols: list, save_path=SHAP_SUMMARY_PATH):
    """
    Generate and save SHAP summary plot for the trained tree-based classifier inside pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    
    # Sample up to 300 rows for fast SHAP computation
    X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42) if len(X_test) > 300 else X_test
    
    # Transform test features
    X_test_trans = preprocessor.transform(X_sample)
    feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)
    
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names)
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_df)
    
    # For binary classification (Random Forest/XGBoost output shape)
    if len(shap_values.shape) == 3:  # (samples, features, classes)
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    # Generate summary plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals.values, X_test_df, show=False)
    plt.title("SHAP Feature Importance (Impact on Churn Prediction)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved SHAP summary plot to: {save_path}")
        
    plt.close("all")
    return explainer, feature_names

def explain_single_prediction(pipeline, explainer, feature_names: list, sample_df: pd.DataFrame):
    """
    Compute SHAP values for a single prediction row for Streamlit visualization.
    """
    preprocessor = pipeline.named_steps["preprocessing"]
    sample_trans = preprocessor.transform(sample_df)
    sample_trans_df = pd.DataFrame(sample_trans, columns=feature_names)
    
    shap_vals = explainer(sample_trans_df)
    
    if len(shap_vals.shape) == 3:
        single_shap = shap_vals[0, :, 1]
    else:
        single_shap = shap_vals[0]
        
    return single_shap, sample_trans_df
