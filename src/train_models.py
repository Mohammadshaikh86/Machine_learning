import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

from src.data_preprocessing import load_and_preprocess_data

def get_models():
    """
    Returns the dictionary of 5 specified classification algorithms.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=9,
            weights='distance',
            metric='minkowski',
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=8,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42
        )
    }

def train_and_evaluate_all():
    """
    Performs 5-Fold Stratified Cross-Validation on training data,
    fits final models, evaluates on hold-out test set, computes financial loss,
    extracts feature importances, and persists models and metrics.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_proc, X_test_proc, preprocessor, feature_names = load_and_preprocess_data()
    
    models = get_models()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    results = {}
    fitted_models = {}
    
    # Financial cost assumptions (per loan in portfolio):
    # False Negative (FN) = Predicted Non-Default, but Actually Defaulted -> Loss of principal (~$10,000)
    # False Positive (FP) = Predicted Default, but Would Repay -> Opportunity cost of lost interest margin (~$1,200)
    # True Positive (TP) = Correctly rejected default -> Loss avoided ($0)
    # True Negative (TN) = Correctly accepted good borrower -> Profit earned (interest margin $1,200)
    COST_FN = 10000.0
    COST_FP = 1200.0
    
    print("\n" + "="*80)
    print("STARTING 5-FOLD CROSS-VALIDATION & MODEL EVALUATION")
    print("="*80)
    
    for name, model in models.items():
        print(f"\n---> Training & Cross-Validating: {name} ...")
        
        # 5-Fold Cross Validation
        cv_scores = cross_validate(model, X_train_proc, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        # Fit final model on full training set
        model.fit(X_train_proc, y_train)
        fitted_models[name] = model
        
        # Predictions on Test set
        y_pred = model.predict(X_test_proc)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_proba = y_pred
            
        # Metrics on hold-out Test set
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate financial cost of errors on test portfolio
        total_financial_loss = (fn * COST_FN) + (fp * COST_FP)
        
        results[name] = {
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_f1": float(f1),
            "test_roc_auc": float(auc),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "financial_loss_usd": float(total_financial_loss),
            "cv_accuracy_mean": float(np.mean(cv_scores['test_accuracy'])),
            "cv_accuracy_std": float(np.std(cv_scores['test_accuracy'])),
            "cv_recall_mean": float(np.mean(cv_scores['test_recall'])),
            "cv_recall_std": float(np.std(cv_scores['test_recall'])),
            "cv_f1_mean": float(np.mean(cv_scores['test_f1'])),
            "cv_f1_std": float(np.std(cv_scores['test_f1'])),
            "cv_roc_auc_mean": float(np.mean(cv_scores['test_roc_auc'])),
            "cv_roc_auc_std": float(np.std(cv_scores['test_roc_auc'])),
            "train_accuracy_mean": float(np.mean(cv_scores['train_accuracy'])),
            "overfitting_gap_acc": float(np.mean(cv_scores['train_accuracy']) - np.mean(cv_scores['test_accuracy']))
        }
        
        print(f"   Accuracy: {acc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(f"   CV ROC-AUC: {np.mean(cv_scores['test_roc_auc']):.4f} (+/- {np.std(cv_scores['test_roc_auc']):.4f})")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"   Total Error Cost: ${total_financial_loss:,.2f}")
    
    # Save all fitted models
    joblib.dump(fitted_models, "models/all_models.joblib")
    
    # Select Best Model based on composite of ROC-AUC and Recall (since Recall is critical for risk)
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_roc_auc'] * 0.5 + results[k]['test_recall'] * 0.5)
    print(f"\n==================================================")
    print(f"🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"==================================================")
    
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, "models/best_model.joblib")
    
    # Feature Importance analysis (using Random Forest and Gradient Boosting)
    feature_importances = {}
    if hasattr(fitted_models["Gradient Boosting"], "feature_importances_"):
        gb_fi = fitted_models["Gradient Boosting"].feature_importances_
        feature_importances["Gradient Boosting"] = dict(zip(feature_names, gb_fi.tolist()))
        
    if hasattr(fitted_models["Random Forest"], "feature_importances_"):
        rf_fi = fitted_models["Random Forest"].feature_importances_
        feature_importances["Random Forest"] = dict(zip(feature_names, rf_fi.tolist()))
        
    # Save results to JSON
    output_summary = {
        "models_evaluation": results,
        "best_model_name": best_model_name,
        "feature_importances": feature_importances,
        "feature_names": feature_names,
        "cost_assumptions": {
            "false_negative_loss_usd": COST_FN,
            "false_positive_loss_usd": COST_FP
        }
    }
    
    with open("models/metrics_comparison.json", "w") as f:
        json.dump(output_summary, f, indent=4)
        
    print("\nSaved all models and metrics to models/metrics_comparison.json")
    return output_summary

if __name__ == "__main__":
    train_and_evaluate_all()
