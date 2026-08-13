# Model Card — E-Commerce Customer Churn Predictor

This Model Card documents the final machine learning model selected for predicting customer churn in an e-commerce platform.

---

## 1. Model Details

- **Model Architecture:** Random Forest Classifier (`n_estimators=200`, `random_state=42`)
- **Pipeline Framework:** `imbalanced-learn` Pipeline containing:
  1. `ColumnTransformer`: `StandardScaler` (numerical) + `OneHotEncoder` (categorical) + `SimpleImputer`
  2. `SMOTE`: Synthetic Minority Over-sampling Technique (`random_state=42`)
  3. `RandomForestClassifier`: Random Forest Ensemble
- **Model Version:** 1.0.0
- **Serialized Artifact:** `models/best_model.joblib` (965 KB)
- **Primary Developer:** Data Science Team

---

## 2. Intended Use

- **Primary Business Goal:** Early detection of e-commerce customers at high risk of churning before churn occurs.
- **Actionable Output:** Real-time risk probability score ($p \in [0, 1]$) used to trigger automated, targeted customer retention campaigns (e.g., personalized cashback vouchers, priority support intervention).
- **Out-of-Scope Uses:** Automated account suspension, dynamic pricing discrimination, or credit scoring.

---

## 3. Training & Validation Strategy

- **Dataset:** 5,630 customer records from E-Commerce Customer Churn dataset.
- **Split Strategy:** 80/20 Stratified Train/Test split (`random_state=42`).
  - Training Set: 4,504 samples (83% non-churn, 17% churn).
  - Test Set: 1,126 samples (held-out, un-smoted).
- **Class Imbalance Handling:** SMOTE applied **strictly to training folds** inside the pipeline to prevent synthetic leakage into test data.

---

## 4. Quantitative Performance Metrics

Evaluated on the 1,126 held-out test samples:

| Model Architecture | ROC-AUC | Recall (Churn) | Precision (Churn) | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Random Forest (Selected)** | **0.9965** | **0.8789** | **0.9435** | **0.9101** |
| XGBoost | 0.9797 | 0.8263 | 0.8920 | 0.8579 |
| Logistic Regression | 0.8970 | 0.8316 | 0.5064 | 0.6295 |

### Confusion Matrix (Random Forest on Held-Out Test Set)

- **True Negatives (Correct Non-Churn):** 923
- **False Positives (False Alarms):** 10
- **False Negatives (Missed Churners):** 23
- **True Positives (Correct Churners):** 167 out of 190

---

## 5. Model Governance & Data Leakage Prevention

| Validation Check | Status | Verification Detail |
|---|:---:|---|
| **SMOTE Leakage Check** | ✅ Passed | SMOTE is embedded inside `ImbPipeline` and executes **after** `train_test_split()`. Held-out test set contains 0 synthetic rows. |
| **Preprocessing Fit Check** | ✅ Passed | `StandardScaler` and `OneHotEncoder` parameters are computed exclusively on `X_train`. |
| **Feature Leakage Check** | ✅ Passed | Target variable `Churn` and surrogate key `CustomerID` are dropped prior to feature transformation. |
| **Reproducibility Check** | ✅ Passed | Global seed `RANDOM_STATE = 42` enforced across splits, SMOTE, and model initializations. |

---

## 6. Model Interpretability & Business Drivers (SHAP)

Global SHAP TreeExplainer analysis highlights the top drivers of customer churn:

1. **`Tenure` & `IsNewCustomer`**: New customers (< 3 months tenure) exhibit significantly higher churn probability.
2. **`Complain`**: Customers filing recent customer service complaints show elevated churn risk.
3. **`DaySinceLastOrder` & `IsInactive`**: Prolonged inactivity (> 15 days) strongly correlates with churn risk.
4. **`CashbackAmount` & `AvgOrderValue`**: Lower monetary incentive / cashback correlates with higher churn likelihood.

---

## 7. Limitations & Recommendations

- **Cross-Sectional Data:** The dataset represents a single temporal snapshot. Temporal validation (out-of-time test set) is recommended when deploying to production with streaming time-series data.
- **Fairness & Demographic Bias:** Features include `Gender` and `MaritalStatus`. Regular bias audits should ensure retention offers are distributed fairly without adverse disparate impact.
- **Model Maintenance Schedule:** Re-evaluate and re-train model quarterly to combat data drift.
