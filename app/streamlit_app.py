import os
import sys
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import shap

from scripts.config import (
    MODEL_PATH, METRICS_PATH, SHAP_SUMMARY_PATH, RAW_DATA_PATH, SAMPLE_DATA_PATH, TARGET_COL
)
from scripts.feature_engineering import add_features, get_column_lists
from scripts.explain import explain_single_prediction, get_transformed_feature_names

# Set Streamlit Page Config
st.set_page_config(
    page_title="Customer Churn Predictor & Governance",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifact():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None

model = load_model_artifact()
metrics = load_metrics()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/line-chart.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🔮 Predict Churn & Explain (SHAP)", "📊 Model Comparison & Governance", "💡 Dataset & Feature Guide"]
)

# --- PAGE 1: PREDICT & EXPLAIN ---
if page == "🔮 Predict Churn & Explain (SHAP)":
    st.markdown('<div class="main-title">🔮 Customer Churn Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter customer demographic and behavioral attributes to calculate real-time churn probability and explain feature impacts using SHAP values.</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model artifact not found! Please run `python run_pipeline.py` first to train and save the model.")
    else:
        st.subheader("📋 Customer Information Input")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0.0, max_value=60.0, value=4.0, step=1.0)
            preferred_login = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
            city_tier = st.selectbox("City Tier", [1, 2, 3], index=2)
            warehouse_to_home = st.number_input("Warehouse to Home (km)", min_value=0.0, max_value=150.0, value=6.0, step=1.0)
            preferred_payment = st.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E wallet", "UPI", "COD", "CC", "Cash on Delivery"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            
        with col2:
            hour_spend = st.number_input("Hours Spend on App", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
            num_device = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=3)
            order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
            satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=2)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            num_address = st.number_input("Number of Addresses", min_value=1, max_value=20, value=9)

        with col3:
            complain = st.selectbox("Filed Complain in Last Month?", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            hike = st.number_input("Order Amount Hike from Last Year (%)", min_value=0.0, max_value=50.0, value=11.0, step=1.0)
            coupon = st.number_input("Coupons Used", min_value=0.0, max_value=30.0, value=1.0, step=1.0)
            order_count = st.number_input("Order Count", min_value=0.0, max_value=50.0, value=1.0, step=1.0)
            days_since_last = st.number_input("Days Since Last Order", min_value=0.0, max_value=60.0, value=5.0, step=1.0)
            cashback = st.number_input("Cashback Amount ($)", min_value=0.0, max_value=500.0, value=159.93, step=10.0)

        predict_btn = st.button("🚀 Predict Churn Probability", type="primary", use_container_width=True)

        if predict_btn:
            raw_input = pd.DataFrame([{
                "Tenure": tenure,
                "PreferredLoginDevice": preferred_login,
                "CityTier": city_tier,
                "WarehouseToHome": warehouse_to_home,
                "PreferredPaymentMode": preferred_payment,
                "Gender": gender,
                "HourSpendOnApp": hour_spend,
                "NumberOfDeviceRegistered": num_device,
                "PreferedOrderCat": order_cat,
                "SatisfactionScore": satisfaction,
                "MaritalStatus": marital_status,
                "NumberOfAddress": num_address,
                "Complain": complain,
                "OrderAmountHikeFromlastYear": hike,
                "CouponUsed": coupon,
                "OrderCount": order_count,
                "DaySinceLastOrder": days_since_last,
                "CashbackAmount": cashback
            }])
            
            # Apply feature engineering
            input_fe = add_features(raw_input)
            
            # Predict
            prob = model.predict_proba(input_fe)[0, 1]
            pred = int(prob >= 0.5)
            
            st.markdown("---")
            st.subheader("🎯 Prediction Result & Action Plan")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric("Churn Risk Probability", f"{prob:.1%}")
                if prob >= 0.5:
                    st.markdown(
                        f'<div class="risk-high">⚠️ HIGH CHURN RISK ({prob:.1%})<br>Status: Likely to Churn</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="risk-low">✅ LOW CHURN RISK ({prob:.1%})<br>Status: Retained / Loyal</div>',
                        unsafe_allow_html=True
                    )

            with res_col2:
                st.markdown("##### 💼 Business Action Recommendation")
                if prob >= 0.5:
                    st.warning(
                        "**High Risk Triggered! Recommended Actions:**\n"
                        "- 🎟️ Assign a $15 personalized cashback coupon immediately.\n"
                        "- 📞 Assign customer support representative if `Complain` == 1.\n"
                        "- 📧 Send targeted re-engagement email workflow."
                    )
                else:
                    st.success(
                        "**Customer is Healthy! Recommended Actions:**\n"
                        "- 🌟 Enroll in loyalty points / VIP tier program.\n"
                        "- 🛍️ Cross-sell complementary products."
                    )

            # SHAP Explanation
            st.markdown("---")
            st.subheader("🔍 SHAP Explanation (Why did the model make this prediction?)")
            
            with st.spinner("Calculating SHAP explanations..."):
                try:
                    preprocessor = model.named_steps["preprocessing"]
                    classifier = model.named_steps["classifier"]
                    num_cols, cat_cols = get_column_lists(input_fe)
                    feature_names = get_transformed_feature_names(preprocessor, num_cols, cat_cols)
                    
                    explainer = shap.TreeExplainer(classifier)
                    single_shap, sample_trans_df = explain_single_prediction(model, explainer, feature_names, input_fe)
                    
                    # Plot waterfall or bar plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(single_shap, max_display=10, show=False)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Could not render SHAP plot: {e}")

# --- PAGE 2: MODEL COMPARISON & GOVERNANCE ---
elif page == "📊 Model Comparison & Governance":
    st.markdown('<div class="main-title">📊 Model Comparison & Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation metrics on held-out test set (1,126 samples) and governance validation checks.</div>', unsafe_allow_html=True)

    if metrics:
        m_df = pd.DataFrame.from_dict(metrics, orient="index")
        st.subheader("🏆 Model Performance Comparison")
        st.dataframe(m_df.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 ROC Curves Comparison")
        if os.path.exists("models/roc_comparison.png"):
            st.image("models/roc_comparison.png", use_container_width=True)
        else:
            st.info("Run `python run_pipeline.py` to generate ROC curves plot.")
            
    with col2:
        st.subheader("🌳 Global SHAP Feature Importance")
        if os.path.exists("models/shap_summary.png"):
            st.image("models/shap_summary.png", use_container_width=True)
        else:
            st.info("Run `python run_pipeline.py` to generate SHAP summary plot.")

    st.markdown("---")
    st.subheader("🛡️ Data Leakage & Validation Checklist")
    st.markdown("""
    - ✅ **Data Splitting**: 80/20 Stratified train/test split performed before any transformation.
    - ✅ **Data Leakage Prevention**: SMOTE and StandardScaler embedded strictly within `imbalanced-learn` pipeline. Transformers fitted *only* on training folds.
    - ✅ **Reproducibility**: Global seed `RANDOM_STATE = 42` enforced across all splits, SMOTE, and model initializations.
    - 📖 **Model Card**: See full governance details in [`MODEL_CARD.md`](./MODEL_CARD.md).
    """)

# --- PAGE 3: DATASET & FEATURE GUIDE ---
else:
    st.markdown('<div class="main-title">💡 Dataset & Feature Engineering Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Overview of underlying features and engineered behavioral indicators.</div>', unsafe_allow_html=True)

    st.subheader("✨ Engineered Features Summary")
    st.markdown("""
    | Feature Name | Calculation Formula | Business Rationale |
    |---|---|---|
    | **`AvgOrderValue`** | `CashbackAmount / (OrderCount + 1)` | Estimates average monetary spend value per order. |
    | **`EngagementScore`** | `HourSpendOnApp + OrderCount + CouponUsed` | Composite score quantifying customer digital engagement level. |
    | **`IsInactive`** | `1 if DaySinceLastOrder > 15 else 0` | Binary warning indicator for recent inactivity. |
    | **`IsNewCustomer`** | `1 if Tenure < 3 else 0` | Binary indicator for newly onboarded customers (< 3 months). |
    """)

    st.subheader("📁 Dataset Summary")
    st.markdown("""
    - **Total Records:** 5,630 customers
    - **Target Variable:** `Churn` (0 = Retained ~83%, 1 = Churned ~17%)
    - **Data Source:** E-Commerce Churn Dataset ([Kaggle Link](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction))
    - **Sample Dataset:** `data/sample.csv` (200-row stratified subset included in repository)
    """)
