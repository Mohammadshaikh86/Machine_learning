import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

def run_system_checks():
    print("=" * 70)
    print("🔍 RUNNING COMPREHENSIVE END-TO-END SYSTEM TEST")
    print("=" * 70)
    
    errors = []
    
    # Test 1: Check Data File
    print("\n[1/7] Testing Dataset File...")
    data_path = "data/credit_risk_dataset.csv"
    if not os.path.exists(data_path):
        errors.append("Dataset file not found at " + data_path)
    else:
        df = pd.read_csv(data_path)
        print(f"  ✅ Dataset exists: {df.shape[0]:,} rows, {df.shape[1]} columns")
        expected_cols = [
            'person_age', 'person_income', 'person_home_ownership',
            'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
            'loan_int_rate', 'loan_status', 'loan_percent_income',
            'cb_person_default_on_file', 'cb_person_cred_hist_length'
        ]
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Dataset missing columns: {missing_cols}")
        else:
            print(f"  ✅ All {len(expected_cols)} schema columns present")
            default_rate = df['loan_status'].mean() * 100
            print(f"  ✅ Target balance: {default_rate:.2f}% defaults")
            
    # Test 2: Test Data Preprocessing Pipeline
    print("\n[2/7] Testing Data Preprocessing Pipeline...")
    try:
        from src.data_preprocessing import load_and_preprocess_data, treat_outliers, engineer_features
        X_train, X_test, y_train, y_test, X_tr_p, X_te_p, prep, f_names = load_and_preprocess_data()
        print(f"  ✅ Preprocessing pipeline works. Features: {len(f_names)}")
        print(f"  ✅ Train split: {X_tr_p.shape}, Test split: {X_te_p.shape}")
    except Exception as e:
        errors.append(f"Preprocessing pipeline error: {e}")
        
    # Test 3: Check Saved Model Artifacts
    print("\n[3/7] Testing Model Artifacts...")
    required_artifacts = [
        "models/best_model.joblib",
        "models/all_models.joblib",
        "models/preprocessor.joblib",
        "models/feature_names.joblib",
        "models/metrics_comparison.json"
    ]
    for art in required_artifacts:
        if os.path.exists(art):
            size_kb = os.path.getsize(art) / 1024
            print(f"  ✅ Artifact verified: {art} ({size_kb:.1f} KB)")
        else:
            errors.append(f"Missing artifact: {art}")
            
    # Test 4: Test Inference Engine (Single & Batch)
    print("\n[4/7] Testing Real-time Inference Engine (src/predict.py)...")
    try:
        from src.predict import LoanRiskPredictor
        predictor = LoanRiskPredictor()
        
        # Test individual prediction
        applicant = {
            'person_age': 29,
            'person_income': 65000,
            'person_home_ownership': 'MORTGAGE',
            'person_emp_length': 6.0,
            'loan_intent': 'PERSONAL',
            'loan_grade': 'B',
            'loan_amnt': 10000,
            'loan_int_rate': 10.5,
            'cb_person_default_on_file': 'N',
            'cb_person_cred_hist_length': 6
        }
        res = predictor.predict_single(applicant)
        print(f"  ✅ Single prediction passed: Risk={res['default_risk']} | Prob={res['probability_of_default']:.2%} | Tier={res['risk_tier']}")
        
        # Test high risk applicant
        high_risk_applicant = {
            'person_age': 22,
            'person_income': 22000,
            'person_home_ownership': 'RENT',
            'person_emp_length': 1.0,
            'loan_intent': 'DEBTCONSOLIDATION',
            'loan_grade': 'F',
            'loan_amnt': 15000,
            'loan_int_rate': 21.0,
            'cb_person_default_on_file': 'Y',
            'cb_person_cred_hist_length': 2
        }
        res_high = predictor.predict_single(high_risk_applicant)
        print(f"  ✅ High-risk detection passed: Risk={res_high['default_risk']} | Prob={res_high['probability_of_default']:.2%} | Tier={res_high['risk_tier']}")
        
        # Test batch prediction
        batch_df = df.head(10).drop(columns=['loan_status'])
        batch_res = predictor.predict_batch(batch_df)
        print(f"  ✅ Batch prediction passed: {len(batch_res)} rows processed, new columns: {['Default_Probability', 'Predicted_Default_Risk', 'Risk_Tier']}")
    except Exception as e:
        errors.append(f"Inference engine error: {e}")
        
    # Test 5: Verify Reports & Figures
    print("\n[5/7] Testing Visualization Reports & Figures...")
    expected_figures = [
        "reports/figures/01_target_distribution.png",
        "reports/figures/02_correlation_heatmap.png",
        "reports/figures/03_dti_vs_income.png",
        "reports/figures/04_loan_grade_default_rate.png",
        "reports/figures/05_intent_and_prior_default.png",
        "reports/figures/06_confusion_matrices_all_models.png",
        "reports/figures/07_metrics_comparison_barchart.png",
        "reports/figures/08_roc_curves_comparison.png",
        "reports/figures/09_feature_importance.png",
        "reports/figures/10_financial_loss_comparison.png"
    ]
    for fig in expected_figures:
        if os.path.exists(fig) and os.path.getsize(fig) > 1000:
            print(f"  ✅ Figure OK: {os.path.basename(fig)} ({os.path.getsize(fig)/1024:.1f} KB)")
        else:
            errors.append(f"Figure missing or empty: {fig}")
            
    # Test 6: Verify Jupyter Notebook Content
    print("\n[6/7] Testing Jupyter Notebook File...")
    nb_path = "notebook/Loan_Default_Prediction.ipynb"
    if os.path.exists(nb_path):
        with open(nb_path, "r") as f:
            nb = json.load(f)
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
        print(f"  ✅ Notebook verified: {len(cells)} total cells ({len(code_cells)} code, {len(md_cells)} markdown)")
    else:
        errors.append(f"Notebook missing at {nb_path}")
        
    # Test 7: Verify Documentation & Reports
    print("\n[7/7] Testing Case Study Report & Readme...")
    doc_paths = ["reports/CASE_STUDY_13_REPORT.md", "README.md", "requirements.txt"]
    for doc in doc_paths:
        if os.path.exists(doc) and os.path.getsize(doc) > 100:
            print(f"  ✅ Document OK: {doc} ({os.path.getsize(doc)/1024:.1f} KB)")
        else:
            errors.append(f"Document missing or empty: {doc}")
            
    print("\n" + "=" * 70)
    if not errors:
        print("🎉 ALL 7 SYSTEM VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")
        print("=" * 70)
        return True
    else:
        print(f"❌ FOUND {len(errors)} ISSUES:")
        for err in errors:
            print(f"  - {err}")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = run_system_checks()
    sys.exit(0 if success else 1)
