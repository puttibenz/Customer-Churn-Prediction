import pandas as pd
from typing import Tuple, List

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering transformations.
    
    Creates 4 new behavioral indicators:
    1. AvgOrderValue: CashbackAmount / (OrderCount + 1)
    2. EngagementScore: HourSpendOnApp + OrderCount + CouponUsed
    3. IsInactive: 1 if DaySinceLastOrder > 15 else 0
    4. IsNewCustomer: 1 if Tenure < 3 else 0
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with engineered features (copy)
    """
    df = df.copy()
    
    # Handle missing values gracefully in calculations if present
    order_count = df["OrderCount"].fillna(0)
    cashback = df["CashbackAmount"].fillna(0)
    hour_spend = df["HourSpendOnApp"].fillna(0)
    coupon = df["CouponUsed"].fillna(0)
    days_since_last = df["DaySinceLastOrder"].fillna(0)
    tenure = df["Tenure"].fillna(0)
    
    df["AvgOrderValue"] = cashback / (order_count + 1)
    df["EngagementScore"] = hour_spend + order_count + coupon
    df["IsInactive"] = (days_since_last > 15).astype(int)
    df["IsNewCustomer"] = (tenure < 3).astype(int)
    
    return df

def get_column_lists(df: pd.DataFrame, drop_cols: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Separate features into numerical and categorical column lists.
    """
    if drop_cols is None:
        drop_cols = ["Churn", "CustomerID"]
        
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    num_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    return num_cols, cat_cols
