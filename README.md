# 📉 Customer Churn Prediction

A complete end-to-end machine learning pipeline to predict customer churn using a high-dimensional real-world dataset.  
This project covers data preprocessing, EDA, model development, rigorous evaluation, and an interactive web app built with Streamlit.



## 📌 Problem Statement

The goal is to accurately predict if a customer will churn, using anonymized behavioral data.

**Dataset Overview:**
- ~167,000 records
- 215 numerical features (`X0` to `X214`)
- Target variable: `Target_ChurnFlag` (0 = Not Churned, 1 = Churned)



## 🔍 Exploratory Data Analysis (EDA)

- **Class Distribution**: 60% retained, 40% churned
- **Missing Data**: Handled via median imputation
- **Noise Removal**: Non-numeric anomalies like "14 month lease" removed
- **Feature Correlation**: Validated with heatmaps and KDE plots


## ⚙️ Modeling Workflow

- **Preprocessing**: Stratified 80:20 split, median imputation
- **Validation**: 5-Fold Cross-Validation
- **Metrics Used**: ROC-AUC (primary), Accuracy, F1 Score, Confusion Matrix



## 🧪 Models Compared

| Model               | Accuracy | F1 Score | ROC-AUC | Remarks               |
|---------------------|----------|----------|---------|------------------------|
| ✅ Random Forest     | 100%     | 1.00     | ~1.00   | Final model            |
| XGBoost              | 100%     | 1.00     | ~0.999  | Close second           |
| Gradient Boosting    | 100%     | 1.00     | ~1.00   | Equally strong         |
| LightGBM             | 100%     | 1.00     | ~1.00   | Slight warnings        |
| Logistic Regression  | ~49%     | 0.49     | ~0.50   | Weak baseline          |


## 🏆 Final Model: Random Forest Classifier

- Training Accuracy: 100%
- Test Accuracy: 100%
- ROC-AUC: ~1.0
- 5-Fold CV ROC-AUC: 0.9999999959
- Confusion Matrix: `[[20020, 0], [1, 13383]]`

---

## 🔬 Model Interpretability

- Top 5 Features: `X16`, `X7`, `X4`, `X19`, `X123`
- Used for UI simplicity — full model trained on all 215 features
- Techniques:
  - Feature Importance (Random Forest)
  - Correlation Heatmap
  - KDE plots

---

## 🚀 Streamlit Web App

A minimal, real-time app to test the model with top 5 input features.

```bash
streamlit run churn_app.py

