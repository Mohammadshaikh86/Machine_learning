import os
import joblib
import numpy as np
import pandas as pd
from src.data_preprocessing import treat_outliers, engineer_features, NUMERICAL_COLS, CATEGORICAL_COLS

class LoanRiskPredictor:
    """
    Production-ready inference pipeline for Loan Default Prediction.
    Supports single-applicant risk scoring, batch loan portfolio assessments,
    and multi-model comparisons.
    """
    def __init__(self, model_name=None):
        try:
            self.preprocessor = joblib.load("models/preprocessor.joblib")
            self.all_models = joblib.load("models/all_models.joblib")
            self.feature_names = joblib.load("models/feature_names.joblib")
        except Exception:
            # If models were pickled with different numpy/scikit-learn versions on deployment host
            from src.train_models import train_and_evaluate_all
            train_and_evaluate_all()
            self.preprocessor = joblib.load("models/preprocessor.joblib")
            self.all_models = joblib.load("models/all_models.joblib")
            self.feature_names = joblib.load("models/feature_names.joblib")
        
        if model_name and model_name in self.all_models:
            self.model_name = model_name
            self.model = self.all_models[model_name]
        else:
            self.model = joblib.load("models/best_model.joblib")
            # Find name of best model
            for k, v in self.all_models.items():
                if type(v) == type(self.model):
                    self.model_name = k
                    break
                    
    def set_model(self, model_name: str):
        if model_name in self.all_models:
            self.model_name = model_name
            self.model = self.all_models[model_name]
            
    def predict_single(self, applicant_data: dict, custom_threshold: float = 0.5) -> dict:
        """
        Takes raw dictionary of applicant attributes and outputs risk assessment.
        """
        df_raw = pd.DataFrame([applicant_data])
        
        # Calculate loan_percent_income if not present
        if 'loan_percent_income' not in df_raw.columns:
            df_raw['loan_percent_income'] = df_raw['loan_amnt'] / df_raw['person_income']
            
        # Treat outliers & engineer features
        df_clean = treat_outliers(df_raw)
        df_feat = engineer_features(df_clean)
        
        # Transform features
        X_input = df_feat[NUMERICAL_COLS + CATEGORICAL_COLS]
        X_proc = self.preprocessor.transform(X_input)
        
        # Probability & Prediction
        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_proc)[0, 1])
        else:
            proba = float(self.model.predict(X_proc)[0])
            
        is_default = bool(proba >= custom_threshold)
        risk_level = "High" if is_default else "Low"
        
        # Risk grade rating
        if proba < 0.15:
            risk_tier = "Prime (Tier 1 - Very Low Risk)"
            recommendation = "Approved with Preferred Interest Rate"
            badge_color = "#28a745"
        elif proba < 0.35:
            risk_tier = "Near-Prime (Tier 2 - Moderate Risk)"
            recommendation = "Approved with Standard Terms"
            badge_color = "#17a2b8"
        elif proba < 0.60:
            risk_tier = "Sub-Prime (Tier 3 - Elevated Risk)"
            recommendation = "Conditional Approval (Requires Higher Collateral / Lower Loan Amount)"
            badge_color = "#ffc107"
        else:
            risk_tier = "High Risk (Tier 4 - Deep Subprime)"
            recommendation = "Decline Loan Application to Avoid Default Loss"
            badge_color = "#dc3545"
            
        # Key Risk Drivers analysis for this specific applicant
        dti = float(df_feat['loan_percent_income'].iloc[0])
        income = float(df_feat['person_income'].iloc[0])
        int_rate = float(df_feat['loan_int_rate'].iloc[0]) if pd.notnull(df_feat['loan_int_rate'].iloc[0]) else 12.0
        hist_def = df_feat['cb_person_default_on_file'].iloc[0]
        
        risk_factors = []
        if dti > 0.35:
            risk_factors.append(f"High Debt-to-Income burden ({dti*100:.1f}% of annual income requested).")
        if hist_def == 'Y':
            risk_factors.append("Historical default record on credit bureau file.")
        if int_rate > 15.0:
            risk_factors.append(f"High assigned interest rate ({int_rate:.1f}%), increasing monthly repayment stress.")
        if income < 35000:
            risk_factors.append(f"Lower income bracket (${income:,.0f}/year).")
            
        if not risk_factors:
            risk_factors.append("Strong financial profile with conservative leverage ratio.")
            
        return {
            "model_used": self.model_name,
            "default_risk": risk_level,
            "probability_of_default": proba,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "badge_color": badge_color,
            "key_risk_factors": risk_factors,
            "dti_ratio": dti,
            "loan_amount": float(df_feat['loan_amnt'].iloc[0]),
            "annual_income": income
        }

    def predict_batch(self, df_raw: pd.DataFrame, custom_threshold: float = 0.5) -> pd.DataFrame:
        """
        Runs batch inference across multiple applicant rows.
        """
        df_copy = df_raw.copy()
        if 'loan_percent_income' not in df_copy.columns:
            df_copy['loan_percent_income'] = df_copy['loan_amnt'] / df_copy['person_income']
            
        df_clean = treat_outliers(df_copy)
        df_feat = engineer_features(df_clean)
        
        X_input = df_feat[NUMERICAL_COLS + CATEGORICAL_COLS]
        X_proc = self.preprocessor.transform(X_input)
        
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_proc)[:, 1]
        else:
            probas = self.model.predict(X_proc)
            
        df_res = df_raw.copy()
        df_res['Default_Probability'] = np.round(probas, 4)
        df_res['Predicted_Default_Risk'] = np.where(probas >= custom_threshold, 'High', 'Low')
        df_res['Risk_Tier'] = pd.cut(
            probas,
            bins=[-0.01, 0.15, 0.35, 0.60, 1.01],
            labels=['Prime', 'Near-Prime', 'Sub-Prime', 'High Risk']
        )
        return df_res

if __name__ == "__main__":
    predictor = LoanRiskPredictor()
    sample_applicant = {
        'person_age': 25,
        'person_income': 48000,
        'person_home_ownership': 'RENT',
        'person_emp_length': 3.0,
        'loan_intent': 'DEBTCONSOLIDATION',
        'loan_grade': 'D',
        'loan_amnt': 18000,
        'loan_int_rate': 16.5,
        'cb_person_default_on_file': 'N',
        'cb_person_cred_hist_length': 4
    }
    result = predictor.predict_single(sample_applicant)
    print("Sample applicant evaluation:")
    print(result)
