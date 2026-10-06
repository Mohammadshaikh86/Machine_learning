import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

NUMERICAL_COLS = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'interest_burden',
    'emp_to_age_ratio'
]

CATEGORICAL_COLS = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

TARGET_COL = 'loan_status'

def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and treats unrealistic outliers in credit data:
    - Age > 100 treated/clipped (biological constraint)
    - Employment length > 60 treated/clipped (career constraint)
    - Extreme income values clipped at 99.5th percentile for model stability
    """
    df_clean = df.copy()
    
    # Age outlier treatment
    df_clean['person_age'] = np.clip(df_clean['person_age'], 18, 90)
    
    # Employment length cannot exceed age - 14
    max_realistic_emp = np.maximum(0, df_clean['person_age'] - 14)
    df_clean['person_emp_length'] = np.minimum(df_clean['person_emp_length'], max_realistic_emp)
    df_clean['person_emp_length'] = np.clip(df_clean['person_emp_length'], 0, 50)
    
    # Cap extreme income at 99.5th percentile to prevent skewing linear models
    income_cap = df_clean['person_income'].quantile(0.995)
    df_clean['person_income'] = np.clip(df_clean['person_income'], 1000, income_cap)
    
    # Ensure loan_percent_income is consistent
    df_clean['loan_percent_income'] = np.round(df_clean['loan_amnt'] / df_clean['person_income'], 4)
    
    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-specific credit risk features:
    - interest_burden: Annual interest payment as % of income
    - emp_to_age_ratio: Stability index of employment over adult lifespan
    """
    df_feat = df.copy()
    
    # Impute temporary median for int_rate if missing just for interaction term
    int_rate = df_feat['loan_int_rate'].fillna(df_feat['loan_int_rate'].median())
    
    # Annual interest burden as a percentage of income
    df_feat['interest_burden'] = np.round((df_feat['loan_amnt'] * (int_rate / 100.0)) / df_feat['person_income'], 4)
    
    # Career stability ratio: employment length relative to adult working years
    emp_len = df_feat['person_emp_length'].fillna(df_feat['person_emp_length'].median())
    adult_years = np.maximum(1.0, df_feat['person_age'] - 18.0)
    df_feat['emp_to_age_ratio'] = np.round(np.clip(emp_len / adult_years, 0.0, 1.0), 4)
    
    return df_feat

def get_preprocessor_pipeline():
    """
    Builds a Scikit-Learn ColumnTransformer pipeline with:
    - Numerical: Median Imputation + StandardScaler
    - Categorical: OneHotEncoder with ignore unknown
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def load_and_preprocess_data(csv_path="data/credit_risk_dataset.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, cleans outliers, engineers features, splits into train/test,
    and fits the preprocessor.
    """
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    
    # 1. Outlier Treatment
    df_clean = treat_outliers(df)
    
    # 2. Feature Engineering
    df_feat = engineer_features(df_clean)
    
    # 3. Features & Target
    X = df_feat[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df_feat[TARGET_COL]
    
    # 4. Stratified Train-Test Split (preserves 21.8% default ratio in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 5. Fit Preprocessor
    preprocessor = get_preprocessor_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Get one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERICAL_COLS + cat_feature_names
    
    # Save preprocessor artifact
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")
    
    print(f"Preprocessing complete. Transformed feature count: {len(feature_names)}")
    print(f"Train set: {X_train_proc.shape}, Test set: {X_test_proc.shape}")
    
    return X_train, X_test, y_train, y_test, X_train_proc, X_test_proc, preprocessor, feature_names

if __name__ == "__main__":
    load_and_preprocess_data()
