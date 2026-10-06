# CASE STUDY 13: Loan Default Prediction Using Machine Learning
**Course:** B.Tech CSE 2024–2028 | Semester V | Machine Learning  
**Subject:** Machine Learning Case Study Project  
**Author:** AI Machine Learning Engineering Team  

---

## 1. Executive Summary & Problem Overview
A Non-Banking Financial Company (NBFC) evaluates loan applications and manages substantial balance sheet exposure. In retail and commercial credit, loan default represents the failure of a borrower to meet legal repayment obligations according to the debt agreement. 

An undetected loan default causes direct principal write-off and legal collection expenses, which heavily outweigh the minor opportunity cost of declining an applicant who might have repaid. This project develops, validates, and deploys an end-to-end Machine Learning credit risk evaluation engine that detects high-risk borrowers prior to loan disbursement and dynamically optimizes loan portfolio profitability.

---

## 2. Dataset Description & Synthetic Generation Methodology

### 2.1 Attribute Schema
The dataset strictly conforms to the industry-standard **Kaggle Credit Risk Dataset**:

| Feature Name | Type | Description | Range / Categories |
| :--- | :--- | :--- | :--- |
| `person_age` | Numerical | Age of the borrower (Years) | 20 to 75 (outliers up to 144 treated) |
| `person_income` | Numerical | Annual personal income ($) | $8,000 to $300,000+ |
| `person_home_ownership` | Categorical | Residential ownership status | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `person_emp_length` | Numerical | Employment tenure (Years) | 0 to 45 (missing values ~8.9%) |
| `loan_intent` | Categorical | Purpose of the loan | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` |
| `loan_grade` | Categorical | Credit risk grade assigned | `A`, `B`, `C`, `D`, `E`, `F`, `G` |
| `loan_amnt` | Numerical | Total requested principal ($) | $500 to $35,000 |
| `loan_int_rate` | Numerical | Assigned interest rate (%) | 5.4% to 23.5% (missing values ~9.5%) |
| `loan_percent_income` | Numerical | Ratio of loan amount to income (DTI) | 0.01 to 0.85 |
| `cb_person_default_on_file` | Categorical | Historical credit bureau default record | `Y` (Yes), `N` (No) |
| `cb_person_cred_hist_length` | Numerical | Credit history length (Years) | 2 to 30 years |
| **`loan_status`** | **Binary Target** | **Target variable** | **0 = Non-Default, 1 = Default (21.8% positive rate)** |

### 2.2 Data Generation Logic
To replicate real-world banking credit conditions:
- **Income Distribution**: Modeled as a Log-Normal distribution ($\mu=10.9, \sigma=0.65$) matching real-world wage skewness.
- **Loan Grade & Pricing**: Graded by risk leverage factor; interest rates conditioned on assigned grades ($A: 7.5\%, \dots, G: 21.5\%$).
- **Structural Default Probability**: Calculated via a structural logistic risk equation incorporating non-linear interactions:
  $$\text{logit}(P) = -3.8 + 5.8(\text{DTI}) + 0.14(\text{IntRate} - 10) + 0.85(\text{HistDefault}) + 0.65(\text{Rent}) + \dots$$
- **Realistic Data Imperfections**: Injected missing values in `person_emp_length` (8.9%) and `loan_int_rate` (9.5%), along with biological/career outliers (age > 100).

---

## 3. Exploratory Data Analysis (EDA) Key Findings

1. **Target Imbalance**: 78.20% non-default (25,478) vs 21.80% default (7,103). Class weighting (`balanced`) is essential during training.
2. **DTI Dominance**: Defaulted borrowers exhibit a median Loan-to-Income ratio of **0.38**, compared to **0.14** for non-defaulting borrowers.
3. **Loan Grade Escalation**: Grade A defaults are $< 5\%$, whereas Grade E/F/G defaults exceed **65%**.
4. **Historical Default Effect**: Applicants with prior bureau default records exhibit an empirical default rate of **52.4%** vs **14.8%** for clean records.

---

## 4. Data Preprocessing, Cleaning & Feature Engineering

### 4.1 Outlier Detection & Treatment
- **Biological & Logical Constraints**: Clipped `person_age` to $[18, 90]$ and constrained `person_emp_length` to $\le \text{age} - 14$.
- **Income Capping**: High-leverage extreme income outliers were capped at the 99.5th percentile to prevent high-gradient distortion in linear models.

### 4.2 Missing Value Imputation
- `person_emp_length`: Imputed with median ($4.0\text{ years}$).
- `loan_int_rate`: Imputed with median ($11.8\%$).
- Imputation was fit strictly inside the scikit-learn `ColumnTransformer` pipeline on the training folds to eliminate **data leakage**.

### 4.3 Feature Engineering
- **Interest Burden**: $\text{interest\_burden} = \frac{\text{loan\_amnt} \times (\text{loan\_int\_rate} / 100)}{\text{person\_income}}$, capturing the fraction of annual salary consumed strictly by interest servicing.
- **Employment-to-Age Ratio**: Measures workforce stability relative to total adult career span.

---

## 5. Machine Learning Algorithms & 5-Fold Cross-Validation

Five distinct classification families were implemented, tuned, and evaluated on a holdout test set ($N=6,517$) with 5-Fold Stratified Cross-Validation:

### Comparative Performance Table

| Algorithm | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | 5-Fold CV ROC-AUC | CV Train-Test Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **95.21%** | **84.51%** | **95.57%** | **89.70%** | **0.9936** | **0.9941 ± 0.0005** | **+0.04% (No Overfit)** |
| **Random Forest (Balanced Subsample)**| 95.47% | 85.77% | 95.00% | 90.15% | 0.9922 | 0.9920 ± 0.0005 | +1.08% |
| **Gradient Boosting** | 96.39% | 91.88% | 91.56% | 91.72% | 0.9932 | 0.9936 ± 0.0004 | +1.79% |
| **Decision Tree (Pruned)** | 94.29% | 81.62% | 95.29% | 87.92% | 0.9881 | 0.9877 ± 0.0018 | +0.61% |
| **KNN ($k=9$)** | 95.27% | 91.01% | 86.91% | 88.91% | 0.9815 | 0.9827 ± 0.0021 | +4.58% |

---

## 6. Confusion Matrix & Asymmetric Financial Loss Analysis

### 6.1 Financial Loss Parameters
- **False Negative (FN) Cost**: $\$10,000$ (Full principal loss on charge-off / bad debt).
- **False Positive (FP) Cost**: $\$1,200$ (Lost interest margin opportunity cost).
- **Cost Ratio**: $\frac{\text{Cost(FN)}}{\text{Cost(FP)}} = \frac{10000}{1200} \approx 8.33 : 1$.

### 6.2 Confusion Matrix Breakdown on Test Set ($N=6,517$)

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Portfolio Loss ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 4,847 | 249 | **63** | 1,358 | **$928,800** *(Lowest Loss)* |
| **Random Forest** | 4,872 | 224 | 71 | 1,350 | **$978,800** |
| **Decision Tree** | 4,791 | 305 | 67 | 1,354 | **$1,036,000** |
| **Gradient Boosting** | 4,981 | 115 | 120 | 1,301 | **$1,338,000** |
| **KNN** | 4,974 | 122 | 186 | 1,235 | **$2,006,400** *(Highest Loss)* |

### Critical Risk Insight:
Although Gradient Boosting achieved the highest raw accuracy (96.39%), **Logistic Regression achieved the highest Recall (95.57%)** and yielded the **lowest total financial loss ($928,800 vs $1,338,000)** because it missed 57 fewer defaults. In credit underwriting, **Recall is the king metric**.

---

## 7. Answers to the 8 Core Project Questions

### Q1: Can loan default be predicted from borrower attributes?
**Yes.** Models demonstrate an empirical ROC-AUC $> 0.993$ and accuracy exceeding $95\%$, proving strong predictive signals in repayment records, interest pricing, and debt ratios.

### Q2: Which attributes are the strongest predictors?
1. **Interest Burden & Debt-to-Income (DTI)** ($\approx 45\%$ importance)
2. **Assigned Interest Rate & Grade** ($\approx 25\%$ importance)
3. **Credit Bureau Prior Default Record** ($\approx 15\%$ importance)

### Q3: Is debt-to-income ratio more informative than income alone?
**Yes, conclusively.** Raw income alone exhibits weak linear correlation ($r=-0.14$) because high earners can easily over-borrow. DTI ($r=+0.38$) measures direct leverage and has $3.2\times$ higher feature importance than raw income.

### Q4: Which algorithm achieves the highest recall?
**Logistic Regression (with Balanced Class Weights)** achieves the highest test recall (**95.57%**), followed by Decision Tree (95.29%) and Random Forest (95.00%).

### Q5: What is the financial cost of each type of error?
- False Negative: Principal write-off ($\approx \$10,000$).
- False Positive: Lost interest margin ($\approx \$1,200$).
- Type II error (FN) is **8.33 times more catastrophic** than Type I error (FP).

### Q6: How well does the model perform on unseen borrowers?
5-Fold Stratified Cross-Validation confirms near-zero variance ($\text{CV ROC-AUC} = 0.9941 \pm 0.0005$). The test set ROC-AUC is $0.9936$, showing negligible overfitting ($< 0.05\%$).

### Q7: Can the model be deployed for portfolio risk monitoring?
**Yes.** The model scores applicants in $< 10\text{ms}$ and provides dynamic risk tiers (`Prime`, `Near-Prime`, `Sub-Prime`, `High Risk`), allowing real-time origination decisions and periodic stress testing.

### Q8: What are the fairness and ethical AI concerns?
- **Disparate Impact**: Younger borrowers with thin credit files must not be rejected outright; lenders should incorporate cash-flow underwriting.
- **FCRA / ECOA Regulatory Compliance**: Automated systems must output adverse action reasons (key risk drivers), which our inference engine provides.
- **Historical Bias Feedback Loops**: Borrowers who repaired past defaults should have positive recent trends weighted appropriately.

---

## 8. Selected Production Model & Justification
**Selected Model:** `Logistic Regression (Class-Balanced)`
- **Recall:** $95.57\%$ (Best-in-class default detection).
- **ROC-AUC:** $0.9936$ (Excellent discrimination).
- **Portfolio Loss:** $\$928,800$ (Lowest portfolio capital depletion).
- **Explainability:** Transparent odds-ratios compliant with financial regulatory auditability.
