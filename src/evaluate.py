import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from src.data_preprocessing import load_and_preprocess_data

def generate_evaluation_visualizations(figures_dir="reports/figures"):
    """
    Generates high-resolution evaluation charts for comparative model study:
    1. Multi-model Confusion Matrices
    2. Benchmark Metrics Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)
    3. Multi-model ROC Curves
    4. Feature Importance Hierarchy (Strongest Predictors)
    5. Financial Loss Comparison per Model
    6. DTI vs Income Importance Comparison
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data and models
    X_train, X_test, y_train, y_test, X_train_proc, X_test_proc, preprocessor, feature_names = load_and_preprocess_data()
    fitted_models = joblib.load("models/all_models.joblib")
    
    with open("models/metrics_comparison.json", "r") as f:
        metrics_data = json.load(f)
        
    model_results = metrics_data["models_evaluation"]
    
    # Set aesthetics
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # 1. Multi-Model Confusion Matrices (2x3 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    model_names = list(fitted_models.keys())
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        model = fitted_models[name]
        y_pred = model.predict(X_test_proc)
        cm = model_results[name]["confusion_matrix"]
        cm_matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
        
        sns.heatmap(cm_matrix, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"{name}\nRecall: {model_results[name]['test_recall']:.2%}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('Actual Label', fontsize=10)
        ax.set_xticklabels(['Non-Default', 'Default'])
        ax.set_yticklabels(['Non-Default', 'Default'])
        
    # Hide the 6th unused subplot
    axes[5].axis('off')
    # Add summary note in 6th plot
    axes[5].text(0.1, 0.5, 
                 "KEY RECALL INSIGHT:\n\n"
                 "• False Negatives (FN) directly cause\n  loan principal loss (~$10,000/default).\n\n"
                 "• False Positives (FP) cause lost\n  interest margin opportunity (~$1,200).\n\n"
                 "• Models with higher Recall\n  significantly minimize total risk.",
                 fontsize=12, bbox=dict(boxstyle="round,pad=1", fc="#f8f9fa", ec="#b0bec5"))
                 
    plt.suptitle("Confusion Matrix Comparison Across All 5 Algorithms", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "06_confusion_matrices_all_models.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. Benchmark Metrics Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------
    metrics_to_plot = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    df_metrics = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": vals["test_accuracy"],
            "Precision": vals["test_precision"],
            "Recall": vals["test_recall"],
            "F1-Score": vals["test_f1"],
            "ROC-AUC": vals["test_roc_auc"]
        }
        for name, vals in model_results.items()
    ])
    
    df_melted = pd.melt(df_metrics, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', palette='tab10', ax=ax, edgecolor='black')
    ax.set_title('Comparative Performance Metrics Across Machine Learning Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
    ax.legend(title='Algorithm', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_metrics_comparison_barchart.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. Multi-Model ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("Set1", len(fitted_models))
    
    for idx, (name, model) in enumerate(fitted_models.items()):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_proba = model.predict(X_test_proc)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=palette[idx], label=f"{name} (AUC = {roc_val:.3f})")
        
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Baseline (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "08_roc_curves_comparison.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Feature Importance (Strongest Predictors)
    # -------------------------------------------------------------
    if "Gradient Boosting" in metrics_data.get("feature_importances", {}):
        gb_fi = metrics_data["feature_importances"]["Gradient Boosting"]
        fi_series = pd.Series(gb_fi).sort_values(ascending=True)
        top_fi = fi_series.tail(12)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_fi.index, top_fi.values, color='#1f77b4', edgecolor='black')
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.3f}', va='center', fontsize=9, fontweight='bold')
        ax.set_title('Strongest Predictors of Loan Default (Gradient Boosting Feature Importance)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Relative Importance (Gini / MDI Reduction)', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "09_feature_importance.png"), dpi=300)
        plt.close()
        
    # -------------------------------------------------------------
    # 5. Financial Loss Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    loss_data = [
        {"Model": name, "Loss": vals["financial_loss_usd"] / 1000.0}
        for name, vals in model_results.items()
    ]
    df_loss = pd.DataFrame(loss_data).sort_values(by="Loss")
    colors = ['#2ca02c' if i == 0 else '#d62728' if i == len(df_loss)-1 else '#1f77b4' for i in range(len(df_loss))]
    bars = ax.bar(df_loss["Model"], df_loss["Loss"], color=colors, edgecolor='black', width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'${h:,.1f}K',
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_title('Financial Loss Impact on Test Loan Portfolio (Lower is Better)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Total Portfolio Loss ($ in Thousands)', fontsize=11)
    ax.set_ylim(0, max(df_loss["Loss"]) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "10_financial_loss_comparison.png"), dpi=300)
    plt.close()
    
    print("All evaluation charts successfully generated and saved to reports/figures/")

if __name__ == "__main__":
    generate_evaluation_visualizations()
