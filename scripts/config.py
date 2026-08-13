import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data paths
RAW_DATA_PATH = DATA_DIR / "E Commerce_Dataset.xlsx"
SAMPLE_DATA_PATH = DATA_DIR / "sample.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary.png"

# Configuration constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Churn"
DROP_COLS = ["CustomerID", TARGET_COL]

# Known numerical & categorical feature columns
NUMERICAL_COLS = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed",
    "OrderCount", "DaySinceLastOrder", "CashbackAmount",
    "AvgOrderValue", "EngagementScore", "IsInactive", "IsNewCustomer"
]

CATEGORICAL_COLS = [
    "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
    "PreferedOrderCat", "MaritalStatus"
]
