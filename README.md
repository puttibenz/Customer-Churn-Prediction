# Customer Churn Prediction (End-to-End ML Pipeline & Streamlit App)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary

Customer churn directly reduces recurring revenue and customer lifetime value in e-commerce. This repository implements a production-grade machine learning pipeline to predict customer churn risk, prevent false positive retention costs, and explain individual churn drivers using **SHAP (SHapley Additive exPlanations)**. Built with scikit-learn and imbalanced-learn pipelines, the final **Random Forest** model achieves a **0.9965 ROC-AUC**, **0.8789 Recall**, and **0.9435 Precision** on a held-out test set (1,126 customers). The repository includes modular code, automated unit tests, model governance documentation, and an interactive Streamlit web application.

---

## 📊 Dataset & Exploratory Data Analysis (EDA) Highlights

- **Total Samples:** 5,630 e-commerce customer records
- **Target Variable:** `Churn` (Binary: `0` = Retained ~83%, `1` = Churned ~17%)
- **Data Source:** [Kaggle E-Commerce Customer Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- **Data Included:** 
  - `data/E Commerce_Dataset.xlsx` (Full original dataset)
  - `data/sample.csv` (200-row stratified sample included for quick testing)

### 🔍 Key EDA Findings & Insights
- **`Tenure` (-0.349 corr):** Strongest churn driver. Customers in their first 0–3 months exhibit the highest churn risk.
- **`Complain` (+0.250 corr):** Customers filing service complaints are significantly more likely to churn.
- **`DaySinceLastOrder` (-0.161 corr):** Inactivity exceeding 15 days indicates an at-risk customer.
- **`CashbackAmount` (-0.154 corr):** Higher cashback rewards directly correlate with customer retention.

> 📖 **Full EDA Documentation:** See [`EDA_SUMMARY.md`](EDA_SUMMARY.md) for detailed statistical distributions, boxplots, correlation matrices, and business retention strategies.

---

## 🏆 Model Performance Comparison

Evaluated on held-out 20% test set ($N = 1,126$):

| Model Architecture | ROC-AUC | Recall (Churn) | Precision (Churn) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.9965** | **0.8789** | **0.9435** | **0.9101** | **Selected Best Model** |
| **XGBoost** | 0.9897 | 0.8263 | 0.8920 | 0.8579 | Candidate |
| **Logistic Regression** | 0.8970 | 0.8316 | 0.5064 | 0.6295 | Baseline |

### Key Business Impact
- **167 out of 190** churn cases accurately identified in the held-out test set.
- **Only 10 False Positives**, minimizing unnecessary marketing / discount voucher costs.
- Enables proactive, targeted customer retention campaigns with high ROI.

---

## 🔮 Interactive Streamlit App

Launch the interactive web application to test predictions and explore SHAP explanations:

```bash
streamlit run app/streamlit_app.py
```

### App Features:
1. **Churn Predictor:** Enter customer attributes $\rightarrow$ real-time risk probability score $\rightarrow$ actionable business recommendation.
2. **SHAP Explanation:** Real-time waterfall plot explaining top features driving individual churn risk up or down.
3. **Model Governance Dashboard:** Compare ROC curves, metrics table, and global feature importance.

---

## 📁 Repository Structure

```
Customer-Churn-Prediction/
├── README.md                      # Executive project summary & documentation
├── MODEL_CARD.md                  # Model governance, leakage checks & card
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned dependencies
├── run_pipeline.py                # End-to-end CLI execution script
├── data/
│   ├── E Commerce_Dataset.xlsx    # Full dataset
│   └── sample.csv                 # 200-row stratified sample for quick runs
├── scripts/                       # Modular Python package
│   ├── __init__.py
│   ├── config.py                  # Paths, constants, feature definitions
│   ├── data_loader.py             # Data loading & schema validation
│   ├── feature_engineering.py     # Domain feature engineering logic
│   ├── preprocessing.py           # ColumnTransformer & ImbPipeline builders
│   ├── train.py                   # Candidate model training & serialization
│   ├── evaluate.py                # Evaluation metrics, ROC & confusion matrix
│   └── explain.py                 # SHAP TreeExplainer & visualization
├── models/                        # Serialized artifacts
│   ├── best_model.joblib          # Trained Random Forest pipeline artifact
│   ├── metrics.json               # Test metrics summary
│   ├── roc_comparison.png         # Saved ROC curves figure
│   └── shap_summary.png           # Saved SHAP summary plot figure
├── app/
│   └── streamlit_app.py           # Interactive Streamlit application
├── tests/                         # Unit test suite
│   ├── __init__.py
│   └── test_feature_engineering.py # Data transform unit tests
└── notebooks/
    └── customer-churn-prediction.ipynb # Exploratory analysis notebook
```

---

## 🛠️ Feature Engineering

Engineered behavioral indicators added to enhance model predictive power:

- **`AvgOrderValue`**: `CashbackAmount / (OrderCount + 1)`
- **`EngagementScore`**: `HourSpendOnApp + OrderCount + CouponUsed`
- **`IsInactive`**: `1 if DaySinceLastOrder > 15 else 0`
- **`IsNewCustomer`**: `1 if Tenure < 3 else 0`

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

```bash
pytest tests/ -v
```

### 3. Run End-to-End ML Pipeline

Run full pipeline on complete dataset:
```bash
python run_pipeline.py
```

Run rapid test on sample dataset:
```bash
python run_pipeline.py --sample
```

### 4. Launch Interactive Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🛡️ Model Governance & Data Leakage Prevention

- **Train/Test Isolation:** 80/20 Stratified split performed before any transformation.
- **Pipeline Encapsulation:** SMOTE oversampling and `StandardScaler` are wrapped inside `imbalanced-learn` pipelines, fitting exclusively on training folds during cross-validation/fit.
- **Full Model Card:** Detailed evaluation, ethics, limitations, and governance documentation are available in [`MODEL_CARD.md`](MODEL_CARD.md).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
