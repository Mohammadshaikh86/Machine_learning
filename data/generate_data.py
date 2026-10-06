import os
import numpy as np
import pandas as pd

def generate_credit_risk_dataset(n_samples=32581, random_state=42, output_path="data/credit_risk_dataset.csv"):
    """
    Generates a realistic Credit Risk Dataset matching the schema, distributions,
    and statistical properties of the Kaggle Credit Risk Dataset.
    
    Attributes:
    - person_age: Age of borrower (20-70)
    - person_income: Annual income ($8,000 - $300,000)
    - person_home_ownership: 'RENT', 'MORTGAGE', 'OWN', 'OTHER'
    - person_emp_length: Employment duration in years (0-40)
    - loan_intent: 'PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'
    - loan_grade: 'A', 'B', 'C', 'D', 'E', 'F', 'G'
    - loan_amnt: Loan amount ($500 - $35,000)
    - loan_int_rate: Interest rate (5.4% - 23.2%)
    - loan_status: Target (0 = Non-Default, 1 = Default, ~21.8% positive rate)
    - loan_percent_income: Ratio of loan amount to annual income (DTI proxy)
    - cb_person_default_on_file: Historical default record ('Y', 'N')
    - cb_person_cred_hist_length: Credit history length in years (2-30)
    """
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Age (Right-skewed, centered around 27)
    age = np.clip(np.random.gamma(shape=9, scale=3, size=n_samples) + 18, 20, 75).astype(int)
    
    # Add a few realistic data anomalies/outliers like in real datasets
    outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.001), replace=False)
    age[outlier_idx[:len(outlier_idx)//2]] = np.random.randint(120, 145, size=len(outlier_idx)//2)
    
    # 2. Employment Length (correlated with age, 0 to 45 years)
    max_emp = np.maximum(0, age - 18)
    emp_length = np.round(np.clip(np.random.exponential(scale=4.5, size=n_samples), 0, max_emp), 1)
    emp_length[outlier_idx[len(outlier_idx)//2:]] = np.random.randint(100, 125, size=len(outlier_idx) - len(outlier_idx)//2)
    
    # 3. Income (Log-normal distribution, median ~$55k, realistic range)
    income_log = np.random.normal(loc=10.9, scale=0.65, size=n_samples)
    income = np.round(np.exp(income_log), -2)
    income = np.clip(income, 8000, 600000)
    # Add high income outliers
    high_inc_idx = np.random.choice(n_samples, size=15, replace=False)
    income[high_inc_idx] = np.random.uniform(1000000, 2500000, size=15)
    
    # 4. Home Ownership
    home_ownership_probs = [0.50, 0.41, 0.08, 0.01]  # RENT, MORTGAGE, OWN, OTHER
    home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=n_samples, p=home_ownership_probs)
    
    # 5. Loan Intent
    intents = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']
    intent_probs = [0.20, 0.19, 0.17, 0.17, 0.16, 0.11]
    loan_intent = np.random.choice(intents, size=n_samples, p=intent_probs)
    
    # 6. Loan Amount ($1,000 to $35,000, modal around $10,000)
    loan_amnt = np.round(np.clip(np.random.gamma(shape=3.5, scale=2800, size=n_samples), 500, 35000), -2).astype(int)
    
    # 7. Loan Percent Income (DTI proxy)
    loan_percent_income = np.round(loan_amnt / income, 2)
    
    # 8. Loan Grade (A to G)
    # Grade depends on loan_percent_income, income, and noise
    risk_factor = (loan_percent_income * 2.5) - (np.log(income) / 15) + np.random.normal(0, 0.35, size=n_samples)
    grade_thresholds = np.percentile(risk_factor, [25, 52, 73, 87, 95, 99])
    
    def assign_grade(rf):
        if rf <= grade_thresholds[0]: return 'A'
        elif rf <= grade_thresholds[1]: return 'B'
        elif rf <= grade_thresholds[2]: return 'C'
        elif rf <= grade_thresholds[3]: return 'D'
        elif rf <= grade_thresholds[4]: return 'E'
        elif rf <= grade_thresholds[5]: return 'F'
        else: return 'G'
        
    loan_grade = np.array([assign_grade(rf) for rf in risk_factor])
    
    # 9. Loan Interest Rate (strongly conditioned on Loan Grade)
    grade_int_base = {'A': 7.5, 'B': 10.8, 'C': 13.2, 'D': 15.6, 'E': 17.8, 'F': 19.5, 'G': 21.5}
    loan_int_rate = np.array([
        np.round(np.random.normal(loc=grade_int_base[g], scale=1.3), 2)
        for g in loan_grade
    ])
    loan_int_rate = np.clip(loan_int_rate, 5.4, 23.5)
    
    # 10. Historical Default on file
    # Default on file correlates with grade and higher interest rate
    p_hist_default = 0.08 + (loan_grade == 'C')*0.05 + (loan_grade == 'D')*0.12 + (loan_grade >= 'E')*0.25
    cb_person_default_on_file = np.where(np.random.rand(n_samples) < p_hist_default, 'Y', 'N')
    
    # 11. Credit History Length (correlated with age, 2 to 30 years)
    cred_hist = np.clip(np.round((age - 18) * np.random.uniform(0.3, 0.8, size=n_samples)), 2, 30).astype(int)
    
    # 12. Target Variable: Loan Status (0 = Non-Default, 1 = Default)
    # Realistic structural credit risk formula
    logit = (
        -3.8 +
        5.8 * loan_percent_income +                          # DTI is strongest predictor
        0.14 * (loan_int_rate - 10) +                        # High interest rate increases default
        0.85 * (cb_person_default_on_file == 'Y') +          # Prior default history
        0.65 * (home_ownership == 'RENT') +                  # Renters have higher default rate
        -0.45 * (home_ownership == 'OWN') +                  # Homeowners have lower default rate
        0.45 * (loan_intent == 'DEBTCONSOLIDATION') +        # High risk intent
        0.35 * (loan_intent == 'MEDICAL') +
        -0.30 * (loan_intent == 'VENTURE') +
        -0.03 * np.minimum(emp_length, 20) +                 # Employment stability reduces risk
        0.55 * (loan_grade == 'D') +
        1.10 * (loan_grade == 'E') +
        1.60 * (loan_grade == 'F') +
        2.10 * (loan_grade == 'G') +
        np.random.normal(0, 0.45, size=n_samples)            # Idiosyncratic risk
    )
    
    prob_default = 1 / (1 + np.exp(-logit))
    # Calibrate to exact standard ~21.8% default rate
    threshold = np.percentile(prob_default, 100 - 21.8)
    loan_status = (prob_default >= threshold).astype(int)
    
    # Introduce realistic missing values (like real Kaggle Credit Risk Dataset)
    # emp_length has ~8.9% missing values, loan_int_rate has ~9.5% missing values
    emp_length_series = pd.Series(emp_length)
    missing_emp_idx = np.random.choice(n_samples, size=int(n_samples * 0.089), replace=False)
    emp_length_series.iloc[missing_emp_idx] = np.nan
    
    int_rate_series = pd.Series(loan_int_rate)
    missing_int_idx = np.random.choice(n_samples, size=int(n_samples * 0.095), replace=False)
    int_rate_series.iloc[missing_int_idx] = np.nan
    
    df = pd.DataFrame({
        'person_age': age,
        'person_income': income,
        'person_home_ownership': home_ownership,
        'person_emp_length': emp_length_series,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amnt': loan_amnt,
        'loan_int_rate': int_rate_series,
        'loan_status': loan_status,
        'loan_percent_income': loan_percent_income,
        'cb_person_default_on_file': cb_person_default_on_file,
        'cb_person_cred_hist_length': cred_hist
    })
    
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} samples with columns: {list(df.columns)}")
    print(f"Default rate: {df['loan_status'].mean()*100:.2f}%")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df

if __name__ == "__main__":
    generate_credit_risk_dataset()
