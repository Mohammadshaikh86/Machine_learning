# 🏦 Case Study 13: Loan Default Prediction Using Machine Learning
> **B.Tech CSE 2024–28 | Semester V | Machine Learning**

---

## 🌟 Project Overview
This project delivers a production-grade **Credit Risk & Loan Default Prediction System** designed for Non-Banking Financial Companies (NBFCs). It leverages statistical modeling, feature engineering, 5-fold cross-validation, and an interactive web application to evaluate loan applicants, detect default risk early, and minimize balance sheet losses.

---

## 🏗️ Project Architecture & Directory Structure
```
ML_main_pro/
│
├── data/
│   ├── generate_data.py          # Synthetic Kaggle Credit Risk Dataset Generator
│   └── credit_risk_dataset.csv   # Dataset (32,581 records, realistic distributions)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Outlier treatment, imputation & feature engineering
│   ├── eda.py                    # Exploratory Data Analysis & plot generation
│   ├── train_models.py           # 5 ML Models, 5-Fold Stratified CV, Cost Matrix
│   ├── evaluate.py               # Comparative ROC, Confusion Matrix, & Loss plots
│   └── predict.py                # Single & Batch inference engine with risk factors
│
├── models/
│   ├── best_model.joblib         # Serialized optimal production model (Logistic Regression)
│   ├── all_models.joblib         # All 5 serialized algorithms
│   ├── preprocessor.joblib       # Fitted ColumnTransformer pipeline
│   ├── feature_names.joblib      # One-hot encoded feature list
│   └── metrics_comparison.json   # Benchmark metrics & cross-validation scores
│
├── reports/
│   ├── figures/                  # 10 publication-quality charts & heatmaps
│   └── CASE_STUDY_13_REPORT.md   # Complete academic & industrial case study report
│
├── notebook/
│   ├── generate_notebook.py
│   └── Loan_Default_Prediction.ipynb # Self-contained Jupyter Notebook for grading
│
├── app.py                        # Premium Streamlit Web Application
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset, Train Models & Evaluate
```bash
# Generate synthetic dataset (32,581 samples)
python3 data/generate_data.py

# Run training, 5-Fold CV, EDA & generate all evaluation plots
export PYTHONPATH=.
python3 src/train_models.py
python3 src/eda.py
python3 src/evaluate.py
```

### 3. Launch Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Summary of Model Comparisons

| Algorithm | Test Accuracy | Test Recall (Default) | Test Precision | Test F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC | Total Financial Loss ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Logistic Regression** | **95.21%** | **95.57%** | **84.51%** | **89.70%** | **0.9936** | **0.9941 ± 0.0005** | **$928,800** |
| 🌲 **Random Forest** | 95.47% | 95.00% | 85.77% | 90.15% | 0.9922 | 0.9920 ± 0.0005 | $978,800 |
| ⚡ **Gradient Boosting** | 96.39% | 91.56% | 91.88% | 91.72% | 0.9932 | 0.9936 ± 0.0004 | $1,338,000 |
| 🌳 **Decision Tree** | 94.29% | 95.29% | 81.62% | 87.92% | 0.9881 | 0.9877 ± 0.0018 | $1,036,000 |
| 📍 **KNN ($k=9$)** | 95.27% | 86.91% | 91.01% | 88.91% | 0.9815 | 0.9827 ± 0.0021 | $2,006,400 |

---

## 💡 Key Research Insights & Answers
1. **Strongest Predictors**: Interest Burden & Debt-to-Income Ratio (DTI) account for $>45\%$ of predictive importance.
2. **DTI vs Income**: DTI is $3.2\times$ more informative than raw income because high earners can easily over-leverage.
3. **Recall Priority**: In banking, a False Negative (undetected default = $\$10,000$ loss) is **8.33x more costly** than a False Positive (lost margin = $\$1,200$). Balanced Logistic Regression achieves the highest Recall (**95.57%**) and saves over **$1,077,000** compared to KNN.
4. **Generalization**: 5-Fold CV shows near-zero overfitting ($<0.05\%$ gap between training and testing).
