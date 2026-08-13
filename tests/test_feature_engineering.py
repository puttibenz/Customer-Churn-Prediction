import pytest
import pandas as pd
import numpy as np
from scripts.feature_engineering import add_features, get_column_lists

@pytest.fixture
def dummy_data():
    return pd.DataFrame({
        "CustomerID": [50001, 50002, 50003, 50004],
        "Churn": [1, 0, 1, 0],
        "Tenure": [2.0, 3.0, 10.0, np.nan],
        "CashbackAmount": [100.0, 200.0, 150.0, np.nan],
        "OrderCount": [1.0, 4.0, 0.0, np.nan],
        "HourSpendOnApp": [3.0, 2.0, 1.0, np.nan],
        "CouponUsed": [1.0, 0.0, 2.0, np.nan],
        "DaySinceLastOrder": [5.0, 15.0, 20.0, np.nan],
        "PreferredLoginDevice": ["Mobile", "Phone", "Computer", "Mobile"]
    })

def test_add_features_adds_four_columns(dummy_data):
    df_transformed = add_features(dummy_data)
    new_cols = ["AvgOrderValue", "EngagementScore", "IsInactive", "IsNewCustomer"]
    for col in new_cols:
        assert col in df_transformed.columns, f"Expected column {col} missing"

def test_add_features_no_mutation(dummy_data):
    original_cols = list(dummy_data.columns)
    add_features(dummy_data)
    assert list(dummy_data.columns) == original_cols, "Input DataFrame was mutated in place"

def test_avg_order_value_calc():
    df = pd.DataFrame({
        "CashbackAmount": [100.0, 200.0],
        "OrderCount": [1.0, 4.0],
        "HourSpendOnApp": [0, 0], "CouponUsed": [0, 0],
        "DaySinceLastOrder": [0, 0], "Tenure": [0, 0]
    })
    transformed = add_features(df)
    # 100 / (1 + 1) = 50.0
    # 200 / (4 + 1) = 40.0
    assert transformed["AvgOrderValue"].iloc[0] == pytest.approx(50.0)
    assert transformed["AvgOrderValue"].iloc[1] == pytest.approx(40.0)

def test_engagement_score_calc():
    df = pd.DataFrame({
        "HourSpendOnApp": [3.0], "OrderCount": [2.0], "CouponUsed": [1.0],
        "CashbackAmount": [0], "DaySinceLastOrder": [0], "Tenure": [0]
    })
    transformed = add_features(df)
    # 3 + 2 + 1 = 6.0
    assert transformed["EngagementScore"].iloc[0] == pytest.approx(6.0)

def test_is_inactive_boundary():
    df = pd.DataFrame({
        "DaySinceLastOrder": [14.0, 15.0, 16.0],
        "CashbackAmount": [0,0,0], "OrderCount": [0,0,0],
        "HourSpendOnApp": [0,0,0], "CouponUsed": [0,0,0], "Tenure": [0,0,0]
    })
    transformed = add_features(df)
    assert transformed["IsInactive"].tolist() == [0, 0, 1]

def test_is_new_customer_boundary():
    df = pd.DataFrame({
        "Tenure": [2.0, 3.0, 4.0],
        "CashbackAmount": [0,0,0], "OrderCount": [0,0,0],
        "HourSpendOnApp": [0,0,0], "CouponUsed": [0,0,0], "DaySinceLastOrder": [0,0,0]
    })
    transformed = add_features(df)
    assert transformed["IsNewCustomer"].tolist() == [1, 0, 0]

def test_add_features_handles_nan(dummy_data):
    transformed = add_features(dummy_data)
    # Row 3 has NaNs — check that calculation returned a valid number (0.0 or binary flag) without raising errors
    assert not np.isnan(transformed["AvgOrderValue"].iloc[3])
    assert not np.isnan(transformed["EngagementScore"].iloc[3])

def test_get_column_lists(dummy_data):
    df_transformed = add_features(dummy_data)
    num_cols, cat_cols = get_column_lists(df_transformed, drop_cols=["Churn", "CustomerID"])
    assert "Tenure" in num_cols
    assert "AvgOrderValue" in num_cols
    assert "PreferredLoginDevice" in cat_cols
    assert "Churn" not in num_cols and "Churn" not in cat_cols
    assert "CustomerID" not in num_cols and "CustomerID" not in cat_cols
