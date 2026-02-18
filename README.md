📊 Customer Churn Prediction

End-to-end Machine Learning pipeline for predicting customer churn using classification models with imbalance handling and business-driven evaluation.

🎯 Project Objective

Customer churn significantly impacts revenue and long-term growth.

The objective of this project is to:

Predict which customers are likely to churn

Handle class imbalance properly

Compare multiple ML models

Select the best model based on business impact

Translate model outputs into actionable retention strategies

📁 Dataset Overview

Binary classification problem (Churn vs Non-Churn)

Imbalanced dataset (~17% churn rate)

1,100+ observations

Behavioral + transactional features

Example features:

Tenure

OrderCount

CashbackAmount

DaySinceLastOrder

HourSpendOnApp

CouponUsed

⚙️ Project Workflow

The project follows a structured ML pipeline:

Phase 1 — Data Preparation

Data cleaning

Missing value handling

Exploratory data analysis

Phase 2 — Feature Engineering

AvgOrderValue

EngagementScore

IsInactive

IsNewCustomer

Behavioral feature combinations

Phase 3 — Modeling Pipeline

Key principles:

Train/Test split before preprocessing

No data leakage

SMOTE applied only on training data

ColumnTransformer for clean preprocessing

Models implemented:

Logistic Regression

Random Forest

XGBoost

📈 Model Evaluation

Evaluation metrics used:

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Threshold tuning

Business cost analysis

🔎 Performance Comparison
Model	ROC-AUC	Recall (Churn)	Precision (Churn)	F1 (Churn)
Logistic Regression	0.89	0.83	0.49	0.61
XGBoost	0.98	0.79	0.88	0.84
Random Forest	0.996	0.87	0.94	0.91
🏆 Selected Model: Random Forest

Selected due to:

Highest ROC-AUC

Strong balance between Recall and Precision

Low False Positives (cost-efficient campaigns)

Low False Negatives (captures most churners)

Confusion Matrix (Random Forest):

True Positives: 166

False Negatives: 24

False Positives: 10

💼 Business Impact

With the selected model:

Detects 87% of churners

Minimizes unnecessary retention campaigns

Reduces churn-related revenue loss

Enables targeted retention strategy

Example business strategy:

Trigger campaign if DaySinceLastOrder > threshold

Focus early engagement on new customers

Reward high-engagement users to increase loyalty

🧠 Advanced Considerations

Class imbalance handled using SMOTE

Threshold optimization based on business cost

Feature importance analysis for interpretability

Pipeline design prevents data leakage

🛠️ Tech Stack

Python

Pandas

Scikit-learn

XGBoost

Imbalanced-learn

Matplotlib / Seaborn
