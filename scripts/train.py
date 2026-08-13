import os
import joblib
import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scripts.config import (
    RANDOM_STATE, MODEL_PATH, MODELS_DIR
)
from scripts.preprocessing import build_preprocessor, build_pipeline

def get_candidate_models() -> Dict[str, Any]:
    """
    Returns dictionary of instantiated baseline algorithms.
    """
    return {
        "Logistic": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        )
    }

def train_models(X_train: pd.DataFrame, y_train: pd.Series, num_cols: list, cat_cols: list) -> Dict[str, Any]:
    """
    Train all candidate models using SMOTE imbalanced-learn pipelines.
    
    Returns
    -------
    Dict[str, ImbPipeline]
        Trained pipelines for each model architecture.
    """
    preprocessor = build_preprocessor(num_cols, cat_cols)
    candidate_classifiers = get_candidate_models()
    trained_pipelines = {}
    
    for name, clf in candidate_classifiers.items():
        print(f"Training {name} pipeline...")
        pipeline = build_pipeline(preprocessor, clf, use_smote=True)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
    return trained_pipelines

def save_model(model: Any, filepath=MODEL_PATH) -> None:
    """
    Save trained model artifact using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Saved model artifact to: {filepath}")

def load_saved_model(filepath=MODEL_PATH) -> Any:
    """
    Load trained model artifact.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at: {filepath}")
    return joblib.load(filepath)
