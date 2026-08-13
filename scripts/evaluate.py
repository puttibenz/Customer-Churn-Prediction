import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
)
from scripts.config import METRICS_PATH

def evaluate_models(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for all trained models on test set.
    """
    metrics_summary = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = float(roc_auc_score(y_test, y_prob))
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1)
        
        metrics_summary[name] = {
            "ROC-AUC": round(auc, 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "F1-Score": round(float(f1), 4)
        }
        
    return metrics_summary

def save_metrics_json(metrics: Dict[str, Dict[str, float]], filepath=METRICS_PATH) -> None:
    """
    Save evaluation metrics summary to JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics summary to: {filepath}")

def plot_roc_curves(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save_path=None):
    """
    Generate ROC curves comparison figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_confusion_matrix(model: Any, X_test: pd.DataFrame, y_test: pd.Series, title="Confusion Matrix"):
    """
    Generate Confusion Matrix plot for a given model.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-Churn (0)", "Churn (1)"],
                yticklabels=["Non-Churn (0)", "Churn (1)"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig
