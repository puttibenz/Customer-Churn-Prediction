# Customer Churn Prediction (End-to-End ML Pipeline)

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict customer churn in an e-commerce business. 

Customer churn directly impacts revenue and long-term growth. The objective of this project is to:
- Identify customers at high risk of churning.
- Minimize missed churn cases (False Negatives).
- Reduce unnecessary retention campaign cost (False Positives).
- Support a data-driven retention strategy.

*The final model achieves near-perfect class separation with strong business interpretability.*

---

## 📊 Dataset Summary
- **Total samples:** 5,630+
- **Target variable:** `Churn` (Binary: 0 = No, 1 = Yes)
- **Class imbalance:** ~83% non-churn / 17% churn
- **Features:** Mixed numerical and categorical features, including behavioral and transactional attributes.

---

## 🧠 Project Workflow

### Phase 1 – Exploratory Data Analysis (EDA)
- Analyzed class imbalance.
- Investigated feature distributions.
- Identified potential churn drivers.
- Checked for missing values and outliers.

### Phase 2 – Feature Engineering
Created additional behavioral indicators to improve model signal strength:
- `AvgOrderValue`
- `EngagementScore`
- `IsInactive`
- `IsNewCustomer`

### Phase 3 – Modeling Pipeline
Implemented a production-style ML pipeline:
- **Data Splitting:** Train/Test split (Stratified).
- **Preprocessing:** `ColumnTransformer` using `StandardScaler` (numerical) and `OneHotEncoder` (categorical).
- **Handling Imbalance:** Applied SMOTE (to training set only).
- **Algorithms Compared:** Logistic Regression, Random Forest, XGBoost.

> **Note:** All preprocessing and SMOTE steps are embedded inside an `imbalanced-learn` pipeline to prevent data leakage.

### Phase 4 – Model Evaluation
Evaluation metrics used: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and ROC Curve comparison.

#### Model Performance Comparison

| Model | ROC-AUC | Recall (Churn) | Precision (Churn) | F1 (Churn) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.890 | 0.83 | 0.49 | 0.61 |
| **Random Forest** | **0.996** | **0.87** | **0.94** | **0.91** |
| **XGBoost** | 0.980 | 0.79 | 0.88 | 0.84 |

---

## 🏆 Final Model Selection
**Selected Model: Random Forest**

**Reasons:**
- Highest ROC-AUC (0.996).
- Best balance between Recall and Precision.
- Only 24 missed churn cases (False Negatives).
- Only 10 false positive predictions.
- Strong business cost efficiency.

---

## 💼 Business Impact
Using the selected model:
- **166 out of 190** churn cases correctly identified.
- Very low false positive rate.
- Enables highly targeted retention campaigns.
- Minimizes marketing waste.
- Improves ROI of customer retention efforts.

*The model can be deployed as an early warning system for churn prevention.*

---

## 🧩 Tech Stack
- **Python**
- **Pandas / NumPy** (Data Manipulation)
- **Scikit-learn** (Machine Learning Pipeline)
- **Imbalanced-learn** (SMOTE)
- **XGBoost** (Advanced Modeling)
- **Matplotlib / Seaborn** (Data Visualization)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
