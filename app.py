import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.predict import LoanRiskPredictor
from src.data_preprocessing import treat_outliers, engineer_features

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinGuard AI · Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* BASE */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f5f0e8 !important;
    color: #1a1a1a !important;
}
.main .block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1.5px solid #e0d8cc !important;
}
section[data-testid="stSidebar"] > div {
    background: #ffffff !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #2a2218 !important;
}

/* TABS — compact, full text */
div[data-baseweb="tab-list"] {
    background: #e8e0d0 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #d5ccbc !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #5c4f3d !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.45rem 0.85rem !important;
    height: auto !important;
    white-space: nowrap !important;
}
div[aria-selected="true"] {
    background: #ffffff !important;
    color: #1a1a1a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
}

/* INPUTS — all text visible */
.stNumberInput label, .stSelectbox label,
.stSlider label, .stTextInput label,
.stFileUploader label {
    color: #2a2218 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}
.stNumberInput input, .stTextInput input {
    background: #fff !important;
    border: 1.5px solid #d5ccbc !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
}
/* ── SELECTBOX — all inner text visible ── */
.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stSelectbox > div > div > div > div,
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div > div {
    background: #ffffff !important;
    border-color: #d5ccbc !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
}
/* Selected value text */
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div[class*="ValueContainer"] *,
.stSelectbox [data-baseweb="select"] div[class*="singleValue"],
.stSelectbox [data-baseweb="select"] div[class*="placeholder"],
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div {
    color: #1a1a1a !important;
}
/* Dropdown list items */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] li {
    color: #1a1a1a !important;
    background: #ffffff !important;
}
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] li:hover {
    background: #f0e8d8 !important;
    color: #1a1a1a !important;
}

/* CONTAINER BORDERS */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1.5px solid #d5ccbc !important;
    border-radius: 14px !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d5ccbc;
    border-radius: 10px;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricValue"] { color: #1a1a1a !important; font-weight:800 !important; }
[data-testid="stMetricLabel"] { color: #7a6a55 !important; font-weight:600 !important; }

/* EXPANDER */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d5ccbc !important;
    border-radius: 12px !important;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary { color: #1a1a1a !important; font-weight:700 !important; }
[data-testid="stExpander"] p { color: #2a2218 !important; line-height:1.7 !important; }

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid #d5ccbc !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* BUTTONS */
.stButton > button {
    background: #1a1a1a !important; color: #f5ede0 !important;
    border: none !important; border-radius: 9px !important;
    font-weight: 700 !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.55rem 1.3rem !important;
}
.stDownloadButton > button {
    background: #f0e8d8 !important; color: #1a1a1a !important;
    border: 1px solid #d5ccbc !important; border-radius: 9px !important;
    font-weight: 600 !important;
}

/* ── Sidebar toggle button — always visible ── */
[data-testid="collapsedControl"] {
    background: #1a1a1a !important;
    border-radius: 0 8px 8px 0 !important;
    color: #f5f0e8 !important;
    width: 28px !important;
    height: 48px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.15) !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="collapsedControl"]:hover {
    background: #333333 !important;
    cursor: pointer !important;
}
[data-testid="collapsedControl"] svg {
    fill: #f5f0e8 !important;
    color: #f5f0e8 !important;
}
/* Also style the sidebar close (collapse) button */
[data-testid="stSidebarCollapseButton"] button,
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"] {
    background: #f5f0e8 !important;
    border: 1px solid #d5ccbc !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebarCollapseButton"] button svg,
button[aria-label="Close sidebar"] svg {
    fill: #1a1a1a !important;
    color: #1a1a1a !important;
}

/* ── Minimal chrome hiding — DO NOT touch header or sidebar controls ── */
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
/* Only hide the deploy button text, not any structural elements */
[data-testid="stDeployButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return LoanRiskPredictor()

@st.cache_data
def load_metrics():
    if not os.path.exists("models/metrics_comparison.json"):
        from src.train_models import train_and_evaluate_all
        train_and_evaluate_all()
    with open("models/metrics_comparison.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/credit_risk_dataset.csv")

predictor    = load_predictor()
metrics_json = load_metrics()
df_data      = load_data()

# ════════════════════════════════════════════════════════════
# HERO HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    background: linear-gradient(130deg,#ede5d0 0%,#e5dac5 60%,#ede5d0 100%);
    border:1.5px solid #d5ccbc; border-radius:20px;
    padding:2rem 2.5rem; margin-bottom:1.5rem; position:relative; overflow:hidden;">

  <!-- decorative circle -->
  <div style="position:absolute;top:-50px;right:-50px;width:220px;height:220px;
    background:radial-gradient(circle,rgba(139,115,85,.15) 0%,transparent 70%);
    border-radius:50%;pointer-events:none;"></div>

  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1.5rem;">
    <!-- Left -->
    <div>
      <div style="font-size:.68rem;font-weight:700;letter-spacing:1.3px;
                  text-transform:uppercase;color:#7a6a55;margin-bottom:.5rem;">
        Case Study 13 &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; B.Tech CSE Sem V
      </div>
      <div style="font-size:2.1rem;font-weight:800;color:#1a1a1a;letter-spacing:-.5px;
                  line-height:1.1;margin-bottom:.4rem;">
        🏦 FinGuard AI
      </div>
      <div style="font-size:.93rem;color:#5c4f3d;margin-bottom:1rem;">
        Credit Risk Assessment &amp; Loan Default Prediction Engine
      </div>
      <div style="display:inline-flex;align-items:center;gap:8px;
                  background:#1a1a1a;color:#f0e8d8;padding:.42rem 1.1rem;
                  border-radius:100px;font-size:.8rem;font-weight:600;">
        <span style="width:8px;height:8px;background:#6ed99f;border-radius:50%;display:inline-block;"></span>
        Model Live &nbsp;·&nbsp; Logistic Regression (Recall-Optimal)
      </div>
    </div>
    <!-- Right stats -->
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
      <div style="background:#fff;border:1px solid #d5ccbc;border-radius:14px;
                  padding:1rem 1.4rem;text-align:center;min-width:120px;">
        <div style="font-size:1.65rem;font-weight:800;color:#1a1a1a;">0.9936</div>
        <div style="font-size:.7rem;font-weight:700;letter-spacing:.5px;
                    text-transform:uppercase;color:#9b8b78;margin-top:3px;">ROC-AUC</div>
      </div>
      <div style="background:#fff;border:1px solid #d5ccbc;border-radius:14px;
                  padding:1rem 1.4rem;text-align:center;min-width:120px;">
        <div style="font-size:1.65rem;font-weight:800;color:#147a45;">95.57%</div>
        <div style="font-size:.7rem;font-weight:700;letter-spacing:.5px;
                    text-transform:uppercase;color:#9b8b78;margin-top:3px;">Recall</div>
      </div>
      <div style="background:#fff;border:1px solid #d5ccbc;border-radius:14px;
                  padding:1rem 1.4rem;text-align:center;min-width:120px;">
        <div style="font-size:1.65rem;font-weight:800;color:#1a1a1a;">32,581</div>
        <div style="font-size:.7rem;font-weight:700;letter-spacing:.5px;
                    text-transform:uppercase;color:#9b8b78;margin-top:3px;">Records</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:.4rem 0 .9rem 0;border-bottom:1px solid #e0d8cc;margin-bottom:.9rem;">
      <div style="font-size:1.05rem;font-weight:800;color:#1a1a1a;">⚙️ Engine Controls</div>
      <div style="font-size:.77rem;color:#7a6a55;margin-top:2px;">Configure the AI engine</div>
    </div>
    """, unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Active ML Algorithm",
        options=list(predictor.all_models.keys()),
        index=0
    )
    predictor.set_model(selected_model)

    risk_threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Probability cut-off: below = Low Risk, above = High Risk"
    )

    st.markdown("""<div style="border-top:1px solid #e0d8cc;margin:1rem 0;"></div>""", unsafe_allow_html=True)

    # Steps
    st.markdown("""
    <div style="font-size:.68rem;font-weight:700;letter-spacing:1.1px;
                text-transform:uppercase;color:#7a6a55;margin-bottom:.7rem;">
      How to Use
    </div>
    """, unsafe_allow_html=True)
    for num, title, desc in [
        ("1","Select Model","Pick an algorithm from the dropdown above"),
        ("2","Enter Details","Fill borrower info in the Predictor tab"),
        ("3","Get Result","See instant risk score and recommendation"),
        ("4","Batch Upload","Score many borrowers via CSV in Batch tab"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:10px;
                    background:#f5f0e8;border:1px solid #e0d8cc;border-radius:10px;
                    padding:.7rem .85rem;margin-bottom:.45rem;">
          <div style="background:#1a1a1a;color:#f0e8d8;width:22px;height:22px;
                      border-radius:50%;display:flex;align-items:center;
                      justify-content:center;font-size:.72rem;font-weight:800;
                      flex-shrink:0;">{num}</div>
          <div>
            <div style="font-size:.84rem;font-weight:700;color:#1a1a1a;">{title}</div>
            <div style="font-size:.76rem;color:#6b5e50;margin-top:1px;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<div style="border-top:1px solid #e0d8cc;margin:1rem 0;"></div>""", unsafe_allow_html=True)

    # Cost model
    st.markdown("""
    <div style="font-size:.68rem;font-weight:700;letter-spacing:1.1px;
                text-transform:uppercase;color:#7a6a55;margin-bottom:.6rem;">
      Portfolio Cost Model
    </div>
    <div style="background:#f5f0e8;border:1px solid #e0d8cc;border-radius:10px;padding:.8rem 1rem;">
      <div style="display:flex;justify-content:space-between;padding:5px 0;
                  border-bottom:1px solid #e0d8cc;font-size:.82rem;color:#2a2218;">
        <span>Missed Default (FN)</span><strong style="color:#b93320;">$10,000</strong>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;
                  border-bottom:1px solid #e0d8cc;font-size:.82rem;color:#2a2218;">
        <span>False Rejection (FP)</span><strong style="color:#147a45;">$1,200</strong>
      </div>
      <div style="display:flex;justify-content:space-between;padding:5px 0;
                  font-size:.82rem;color:#2a2218;">
        <span>FN : FP Ratio</span><strong style="color:#1a1a1a;">8.33 : 1</strong>
      </div>
    </div>
    <div style="font-size:.74rem;color:#7a6a55;margin-top:.6rem;line-height:1.55;">
      💡 Maximising <strong style="color:#2a2218;">Recall</strong> directly saves capital —
      each missed default costs <strong style="color:#b93320;">8× more</strong> than a false rejection.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predictor",
    "📊 Models",
    "📈 Analysis",
    "🧠 Q&A",
    "📁 Batch"
])

# helper: section chip
def chip(text):
    st.markdown(f"""
    <div style="display:inline-block;font-size:.67rem;font-weight:700;letter-spacing:1.1px;
                text-transform:uppercase;color:#7a6a55;background:#e8e0d0;
                border:1px solid #d5ccbc;padding:3px 11px;border-radius:100px;
                margin-bottom:.85rem;">{text}</div>
    """, unsafe_allow_html=True)

# helper: col header
def col_header(icon, text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:.85rem;
                padding-bottom:.6rem;border-bottom:1px solid #e8e0d0;">
      <span style="font-size:1.1rem;">{icon}</span>
      <span style="font-size:.9rem;font-weight:700;color:#1a1a1a;">{text}</span>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 1 — RISK PREDICTOR
# ════════════════════════════════════════════════════════════
with tab1:
    chip("Fill borrower details · results update instantly")

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            col_header("👤", "Personal & Employment")
            person_age            = st.number_input("Age (Years)", min_value=18, max_value=85, value=28, step=1)
            person_income         = st.number_input("Annual Income ($)", min_value=8000, max_value=500000, value=52000, step=1000)
            person_emp_length     = st.number_input("Employment Length (Years)", min_value=0.0, max_value=45.0, value=4.0, step=0.5)
            person_home_ownership = st.selectbox("Home Ownership", ['RENT','MORTGAGE','OWN','OTHER'], index=0)

    with c2:
        with st.container(border=True):
            col_header("💳", "Loan Details")
            loan_amnt     = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=12000, step=500)
            loan_intent   = st.selectbox("Loan Purpose",
                ['PERSONAL','EDUCATION','MEDICAL','VENTURE','HOMEIMPROVEMENT','DEBTCONSOLIDATION'], index=5)
            loan_grade    = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'], index=3)
            loan_int_rate = st.slider("Interest Rate (%)", 5.0, 24.0, 14.5, 0.1)

    with c3:
        with st.container(border=True):
            col_header("📜", "Credit History")
            cb_default  = st.selectbox("Prior Default on Record?", ['N','Y'], index=0)
            cb_hist_len = st.number_input("Credit History Length (Years)", min_value=2, max_value=35, value=5, step=1)

            st.markdown("""<div style="border-top:1px solid #e8e0d0;margin:.9rem 0;"></div>""", unsafe_allow_html=True)
            col_header("📐", "Live Risk Indicators")

            calc_dti = loan_amnt / max(1.0, person_income)
            ann_int  = loan_amnt * (loan_int_rate / 100.0)
            dti_col  = "#b93320" if calc_dti > 0.35 else "#147a45"
            dti_tag  = "⚠ High" if calc_dti > 0.35 else "✓ Safe"

            st.markdown(f"""
            <div style="background:#f5f0e8;border:1.5px solid #d5ccbc;border-radius:9px;
                        padding:.65rem 1rem;margin-bottom:8px;
                        display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:.81rem;font-weight:600;color:#5c4f3d;">Debt-to-Income (DTI)</span>
              <span style="font-weight:800;color:{dti_col};">{calc_dti*100:.1f}%
                <span style="font-size:.71rem;opacity:.85;">&nbsp;{dti_tag}</span>
              </span>
            </div>
            <div style="background:#f5f0e8;border:1.5px solid #d5ccbc;border-radius:9px;
                        padding:.65rem 1rem;
                        display:flex;justify-content:space-between;align-items:center;">
              <span style="font-size:.81rem;font-weight:600;color:#5c4f3d;">Annual Interest Cost</span>
              <span style="font-weight:800;color:#1a1a1a;">${ann_int:,.0f}/yr</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Predict ──────────────────────────────────────────────
    pred_res = predictor.predict_single({
        'person_age': person_age, 'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_intent': loan_intent, 'loan_grade': loan_grade,
        'loan_amnt': loan_amnt, 'loan_int_rate': loan_int_rate,
        'cb_person_default_on_file': cb_default,
        'cb_person_cred_hist_length': cb_hist_len
    }, custom_threshold=risk_threshold)

    prob = pred_res["probability_of_default"]
    is_high = pred_res["default_risk"] == "High"

    st.markdown("""<div style="border-top:1.5px solid #d5ccbc;margin:1.5rem 0 1rem;"></div>""", unsafe_allow_html=True)
    chip("Assessment Result")

    r1, r2 = st.columns([1.3, 1], gap="medium")

    with r1:
        g_col = "#b93320" if is_high else "#147a45"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x':[0,1],'y':[0,1]},
            title={'text': f"Default Probability · {selected_model}",
                   'font': {'size':13,'color':'#5c4f3d','family':'Plus Jakarta Sans'}},
            number={'suffix':"%",'valueformat':".1f",
                    'font':{'size':50,'color':g_col,'family':'Plus Jakarta Sans'}},
            gauge={
                'axis':{'range':[None,100],'tickwidth':1,
                        'tickcolor':'#c0b8aa','tickfont':{'color':'#9b8b78','size':10}},
                'bar':{'color':g_col,'thickness':.22},
                'bgcolor':'#f5f0e8','borderwidth':0,
                'steps':[
                    {'range':[0,25], 'color':'#d4f0e0'},
                    {'range':[25,50],'color':'#fdefc8'},
                    {'range':[50,75],'color':'#fddcc0'},
                    {'range':[75,100],'color':'#fdc8c4'}
                ],
                'threshold':{'line':{'color':'#5c4f3d','width':3},
                             'thickness':.75,'value':risk_threshold*100}
            }
        ))
        fig.update_layout(
            height=265, margin=dict(l=15,r=15,t=55,b=10),
            paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
            font=dict(family='Plus Jakarta Sans')
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        with st.container(border=True):
            badge_bg  = "#fdf0ee" if is_high else "#eaf7f0"
            badge_col = "#b93320" if is_high else "#147a45"
            badge_bdr = "#f0a095" if is_high else "#96dbb8"
            badge_ico = "⚠️" if is_high else "✅"
            badge_txt = "HIGH RISK" if is_high else "LOW RISK"

            st.markdown(f"""
            <div style="display:inline-flex;align-items:center;gap:8px;
                        background:{badge_bg};color:{badge_col};
                        border:1.5px solid {badge_bdr};
                        padding:.65rem 1.5rem;border-radius:100px;
                        font-weight:800;font-size:1.05rem;margin-bottom:1rem;">
              {badge_ico} &nbsp;{badge_txt} &nbsp;·&nbsp; {prob*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

            for lbl, val, pill in [
                ("Risk Tier",       pred_res['risk_tier'],       False),
                ("Recommendation",  pred_res['recommendation'],  False),
                ("Model",           selected_model,              True),
                ("Threshold",       f"{risk_threshold:.2f}",     True),
            ]:
                val_html = (f'<span style="background:#f0e8d8;color:#3d3530;'
                            f'padding:2px 11px;border-radius:100px;'
                            f'font-size:.79rem;font-weight:600;">{val}</span>'
                            if pill else
                            f'<span style="font-weight:700;color:#1a1a1a;font-size:.87rem;">{val}</span>')
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:8px 0;border-bottom:1px solid #ede5d8;">
                  <span style="font-size:.81rem;color:#9b8b78;">{lbl}</span>
                  {val_html}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:.75rem;font-weight:700;color:#7a6a55;
                        letter-spacing:.6px;margin-top:.9rem;margin-bottom:.4rem;">
              KEY RISK FACTORS
            </div>
            """, unsafe_allow_html=True)
            pills = " ".join([
                f'<span style="display:inline-block;background:#f0e8d8;border:1px solid #d5ccbc;'
                f'color:#3d3530;padding:3px 11px;border-radius:100px;font-size:.78rem;'
                f'font-weight:500;margin:2px;">{f}</span>'
                for f in pred_res["key_risk_factors"]
            ])
            st.markdown(f'<div style="line-height:2;">{pills}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
with tab2:
    chip("5 algorithms · Holdout test set · 6,517 samples")

    rows = []
    for m, mm in metrics_json["models_evaluation"].items():
        cm = mm["confusion_matrix"]
        rows.append({
            "Algorithm":       m,
            "Accuracy":        f"{mm['test_accuracy']:.2%}",
            "Precision":       f"{mm['test_precision']:.2%}",
            "Recall ↑":        f"{mm['test_recall']:.2%}",
            "F1-Score":        f"{mm['test_f1']:.2%}",
            "ROC-AUC":         f"{mm['test_roc_auc']:.4f}",
            "CV ROC-AUC":      f"{mm['cv_roc_auc_mean']:.4f} ± {mm['cv_roc_auc_std']:.4f}",
            "Missed Defaults": cm["FN"],
            "Portfolio Loss":  f"${mm['financial_loss_usd']:,.0f}"
        })
    st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), use_container_width=True)

    st.markdown("""
    <div style="background:#f0e8d8;border-left:4px solid #8b7355;
                padding:1rem 1.3rem;border-radius:0 10px 10px 0;
                font-size:.87rem;color:#2a2218;margin:1.2rem 0;line-height:1.65;">
      💡 <strong>Why Logistic Regression wins:</strong> Gradient Boosting has the highest accuracy (96.39%)
      but misses <strong>120 defaults</strong> → $1,338,000 loss. Logistic Regression misses only
      <strong>63 defaults</strong> → saving <strong>$409,200</strong> in capital.
      In credit risk, <strong>Recall is the king metric</strong>.
    </div>
    """, unsafe_allow_html=True)

    bc1, bc2 = st.columns(2, gap="medium")
    with bc1:
        chip("ROC Curves — All Models")
        if os.path.exists("reports/figures/08_roc_curves_comparison.png"):
            st.image("reports/figures/08_roc_curves_comparison.png", use_container_width=True)
        else:
            st.info("Run `python3 -m src.evaluate` to generate charts.")
    with bc2:
        chip("Portfolio Financial Loss")
        if os.path.exists("reports/figures/10_financial_loss_comparison.png"):
            st.image("reports/figures/10_financial_loss_comparison.png", use_container_width=True)

    chip("Confusion Matrices — All 5 Models")
    if os.path.exists("reports/figures/06_confusion_matrices_all_models.png"):
        st.image("reports/figures/06_confusion_matrices_all_models.png", use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — DATA ANALYSIS
# ════════════════════════════════════════════════════════════
with tab3:
    chip("Exploratory Data Analysis · 32,581 borrower records")

    ec1, ec2 = st.columns(2, gap="medium")
    with ec1:
        chip("Target Distribution (Class Imbalance)")
        if os.path.exists("reports/figures/01_target_distribution.png"):
            st.image("reports/figures/01_target_distribution.png", use_container_width=True)
        chip("DTI vs Income Scatter")
        if os.path.exists("reports/figures/03_dti_vs_income.png"):
            st.image("reports/figures/03_dti_vs_income.png", use_container_width=True)
    with ec2:
        chip("Correlation Heatmap")
        if os.path.exists("reports/figures/02_correlation_heatmap.png"):
            st.image("reports/figures/02_correlation_heatmap.png", use_container_width=True)
        chip("Loan Grade & Default Rate")
        if os.path.exists("reports/figures/04_loan_grade_default_rate.png"):
            st.image("reports/figures/04_loan_grade_default_rate.png", use_container_width=True)

    st.markdown("""<div style="border-top:1.5px solid #d5ccbc;margin:1.5rem 0 1rem;"></div>""", unsafe_allow_html=True)
    chip("Interactive · Default Rate by Home Ownership")

    own = df_data.groupby('person_home_ownership')['loan_status'].mean().reset_index()
    own.columns = ['Home Ownership','Default Rate']
    fig_own = px.bar(own, x='Home Ownership', y='Default Rate',
        color='Default Rate',
        color_continuous_scale=[[0,'#c8f0d8'],[.5,'#fde8a8'],[1,'#f5b0a8']],
        template='plotly_white', title='Default Rate by Home Ownership Type')
    fig_own.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#faf6f0',
        font=dict(family='Plus Jakarta Sans',color='#1a1a1a',size=12),
        height=300, margin=dict(l=10,r=10,t=40,b=10),
        showlegend=False, coloraxis_showscale=False,
        title_font=dict(size=13,color='#1a1a1a'))
    fig_own.update_traces(marker_line_width=0)
    st.plotly_chart(fig_own, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — Q&A
# ════════════════════════════════════════════════════════════
with tab4:
    chip("8 Research Questions · Evidence-Backed Answers")

    qa = [
        ("Q1 · Can loan default be predicted from borrower attributes?",
         "Yes — ROC-AUC > 0.993 and ~95.5% accuracy. DTI, loan grade, interest rate, and prior default history carry strong predictive signal. Machine learning decisively outperforms manual credit scoring."),
        ("Q2 · Which attributes are the strongest predictors?",
         "① Interest Burden + DTI ≈ 45% importance. ② Interest Rate + Loan Grade ≈ 25%. ③ Prior Bureau Default ≈ 15%. Engineered features `interest_burden` and `emp_to_age_ratio` added significant predictive lift.", True),
        ("Q3 · Is DTI more informative than raw income?",
         "Yes. Income alone: r = −0.14 (weak). DTI: r = +0.38 with 3.2× higher feature importance. DTI measures actual repayment leverage — income alone doesn't reveal whether debt load is sustainable."),
        ("Q4 · Which algorithm achieves the highest recall?",
         "Logistic Regression (Balanced) → 95.57% Recall. Caught 1,358 of 1,421 actual defaults, missing only 63. Decision Tree (95.29%) and Random Forest (95.00%) follow."),
        ("Q5 · What is the financial cost of each error type?",
         "False Negative (missed default) = $10,000 principal loss. False Positive (false rejection) = $1,200 opportunity cost. FN is 8.33× more expensive. Maximising Recall directly minimises portfolio capital damage."),
        ("Q6 · How well does the model generalise to unseen borrowers?",
         "5-Fold CV ROC-AUC = 0.9941 ± 0.0005. Test ROC-AUC = 0.9936. Overfitting gap < 0.05%. Near-zero variance confirms reliable generalisation to new applicants."),
        ("Q7 · Can it be deployed for portfolio monitoring?",
         "Yes. Inference < 10ms. Risk tiers (Prime / Near-Prime / Sub-Prime / High Risk) enable automated pricing, real-time origination decisions, and periodic portfolio stress testing."),
        ("Q8 · Fairness & Ethical AI concerns?",
         "① Disparate impact on young/thin-file borrowers → supplement with cash-flow data. ② FCRA/ECOA compliance → model outputs human-readable adverse action reasons. ③ Historical bias loops → weight recent positive repayment behaviour.")
    ]

    for item in qa:
        show_img = len(item) > 2 and item[2]
        with st.expander(item[0]):
            st.write(item[1])
            if show_img and os.path.exists("reports/figures/09_feature_importance.png"):
                st.image("reports/figures/09_feature_importance.png", use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — BATCH UPLOAD
# ════════════════════════════════════════════════════════════
with tab5:
    chip("Upload a CSV to score multiple borrowers at once")

    b1, b2 = st.columns([1, 1.6], gap="medium")

    with b1:
        with st.container(border=True):
            col_header("📋", "How Batch Scoring Works")
            for num, title, desc in [
                ("1","Download Template","Get sample CSV with correct columns"),
                ("2","Fill Your Data",   "Add your borrower rows"),
                ("3","Upload & Score",   "Drop CSV into the uploader"),
                ("4","Download Results","Get scored CSV with risk tiers"),
            ]:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:9px;
                            background:#f5f0e8;border:1px solid #e0d8cc;border-radius:9px;
                            padding:.65rem .85rem;margin-bottom:.4rem;">
                  <div style="background:#1a1a1a;color:#f0e8d8;width:22px;height:22px;
                              border-radius:50%;display:flex;align-items:center;justify-content:center;
                              font-size:.72rem;font-weight:800;flex-shrink:0;">{num}</div>
                  <div>
                    <div style="font-size:.83rem;font-weight:700;color:#1a1a1a;">{title}</div>
                    <div style="font-size:.75rem;color:#6b5e50;margin-top:1px;">{desc}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        sample_csv = df_data.drop(columns=['loan_status']).head(20).to_csv(index=False).encode()
        st.download_button("📥 Download Sample Template",
            data=sample_csv, file_name="sample_applicants.csv",
            mime="text/csv", use_container_width=True)

    with b2:
        uploaded = st.file_uploader("Upload Applicants CSV", type=["csv"])

        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded **{len(batch_df)}** applicant records")

            with st.spinner("Scoring portfolio..."):
                results = predictor.predict_batch(batch_df, custom_threshold=risk_threshold)

            high  = (results['Predicted_Default_Risk']=='High').sum()
            low   = (results['Predicted_Default_Risk']=='Low').sum()
            avg_p = results['Default_Probability'].mean()*100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",        len(results))
            m2.metric("High Risk",    high, delta=f"{high/len(results)*100:.0f}%", delta_color="inverse")
            m3.metric("Low Risk",     low)
            m4.metric("Avg Prob",     f"{avg_p:.1f}%")

            fig_t = px.histogram(results, x="Risk_Tier", color="Risk_Tier",
                title="Portfolio Risk Tier Distribution",
                color_discrete_map={'Prime':'#2ecc71','Near-Prime':'#3498db',
                                    'Sub-Prime':'#e8a820','High Risk':'#e74c3c'},
                template='plotly_white')
            fig_t.update_layout(
                paper_bgcolor='#ffffff', plot_bgcolor='#f5f0e8',
                font=dict(family='Plus Jakarta Sans',color='#1a1a1a'),
                height=270, margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False, title_font=dict(size=13,color='#1a1a1a'))
            fig_t.update_traces(marker_line_width=0)
            st.plotly_chart(fig_t, use_container_width=True)

            chip("Scored Results Table")
            st.dataframe(results, use_container_width=True)

            st.download_button("📥 Download Scored Results",
                data=results.to_csv(index=False).encode(),
                file_name="scored_portfolio.csv", mime="text/csv")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3.5rem 1rem;background:#ffffff;
                        border:2px dashed #d5ccbc;border-radius:16px;">
              <div style="font-size:2.5rem;margin-bottom:.6rem;">📄</div>
              <div style="font-size:.95rem;font-weight:700;color:#2a2218;">Drop your CSV here</div>
              <div style="font-size:.8rem;color:#9b8b78;margin-top:.3rem;">
                Download the template first → fill it → upload here
              </div>
            </div>
            """, unsafe_allow_html=True)
