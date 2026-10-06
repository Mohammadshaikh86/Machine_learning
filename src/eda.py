import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def perform_eda(data_path="data/credit_risk_dataset.csv", figures_dir="reports/figures"):
    """
    Executes full Exploratory Data Analysis and saves publication-quality charts.
    """
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Set global aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    print("=== DATASET SUMMARY ===")
    print(df.info())
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())
    
    # 1. Target Class Distribution (Imbalance)
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['loan_status'].value_counts()
    labels = ['Non-Default (0)', 'Default (1)']
    colors = ['#2b5c8f', '#d9534f']
    bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor='black', alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f'{height:,}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    ax.set_title('Loan Default Target Class Distribution (Imbalance Analysis)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Borrowers', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '01_target_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap of Numerical Features
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'loan_status']
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=1, square=True, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title('Correlation Heatmap of Financial & Credit Attributes', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Debt-To-Income (loan_percent_income) vs Income Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # DTI Boxplot
    sns.boxplot(x='loan_status', y='loan_percent_income', data=df, ax=axes[0], palette=['#4575b4', '#d73027'])
    axes[0].set_title('Loan-to-Income Ratio (DTI) by Default Status', fontweight='bold', fontsize=12)
    axes[0].set_xticklabels(['Non-Default (0)', 'Default (1)'])
    axes[0].set_ylabel('Loan Amount as % of Annual Income')
    axes[0].set_xlabel('Loan Status')
    
    # Income Distribution KDE (log scale)
    sns.kdeplot(data=df, x='person_income', hue='loan_status', common_norm=False,
                log_scale=True, ax=axes[1], palette=['#4575b4', '#d73027'], fill=True, alpha=0.3)
    axes[1].set_title('Annual Income Distribution (Log Scale) by Default Status', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Annual Income ($)')
    axes[1].legend(title='Loan Status', labels=['Default', 'Non-Default'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_dti_vs_income.png'), dpi=300)
    plt.close()
    
    # 4. Loan Grade & Default Rates
    fig, ax = plt.subplots(figsize=(9, 5))
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_default = df.groupby('loan_grade')['loan_status'].mean().reindex(grade_order) * 100
    grade_bars = ax.bar(grade_default.index, grade_default.values, color=sns.color_palette("YlOrRd", len(grade_order)), edgecolor='black')
    for bar in grade_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_title('Observed Default Rate by Loan Grade (A to G)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Assigned Loan Grade', fontsize=11)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_loan_grade_default_rate.png'), dpi=300)
    plt.close()
    
    # 5. Loan Intent & Historical Default on File
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # By Loan Intent
    intent_def = df.groupby('loan_intent')['loan_status'].mean().sort_values(ascending=False) * 100
    axes[0].barh(intent_def.index, intent_def.values, color='#2c7bb6', edgecolor='black')
    for i, v in enumerate(intent_def.values):
        axes[0].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    axes[0].set_title('Default Rate by Loan Intent', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Default Rate (%)')
    
    # By Historical Default on File
    hist_def = df.groupby('cb_person_default_on_file')['loan_status'].mean() * 100
    axes[1].bar(['No Prior Default (N)', 'Prior Default on File (Y)'], hist_def.values, color=['#74add1', '#f46d43'], edgecolor='black', width=0.45)
    for i, v in enumerate(hist_def.values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    axes[1].set_title('Default Rate by Credit Bureau Historical Record', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].set_ylim(0, 70)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_intent_and_prior_default.png'), dpi=300)
    plt.close()
    
    print(f"All EDA figures successfully saved to {figures_dir}")

if __name__ == "__main__":
    perform_eda()
