"""
==================================================================
 STUDENT PLACEMENT PREDICTION SYSTEM - WEB APP (Streamlit)
==================================================================
Run locally with:
    pip install streamlit scikit-learn pandas numpy
    streamlit run app.py

Deploy for free at: https://share.streamlit.io  (Streamlit Community Cloud)
==================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = [
    "CGPA", "IQ", "Internships", "Projects_Completed",
    "Communication_Skill", "Backlogs", "Extra_Curricular", "Aptitude_Score"
]

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)


# ------------------------------------------------------------------
# DATA + MODEL (cached so it only trains once, not on every click)
# ------------------------------------------------------------------
@st.cache_data
def generate_dataset(n_samples=1200):
    cgpa = np.round(np.random.normal(7.0, 1.1, n_samples).clip(4, 10), 2)
    iq = np.round(np.random.normal(105, 12, n_samples).clip(80, 140), 0)
    internships = np.random.poisson(1.0, n_samples).clip(0, 3)
    projects = np.random.poisson(2.5, n_samples).clip(0, 6)
    communication = np.round(np.random.normal(6.5, 1.5, n_samples).clip(1, 10), 1)
    backlogs = np.random.poisson(0.6, n_samples).clip(0, 4)
    extra_curricular = np.round(np.random.normal(5.0, 2.0, n_samples).clip(0, 10), 1)
    aptitude = np.round(np.random.normal(65, 15, n_samples).clip(0, 100), 1)

    score = (
        cgpa * 6.5 + (iq - 80) * 0.35 + internships * 8 + projects * 4
        + communication * 5 + extra_curricular * 2 - backlogs * 10 + aptitude * 0.4
    )
    prob = 1 / (1 + np.exp(-(score - np.mean(score)) / (np.std(score) * 0.9)))
    noise = np.random.normal(0, 0.07, n_samples)
    prob = np.clip(prob + noise, 0, 1)
    placed = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "CGPA": cgpa, "IQ": iq, "Internships": internships,
        "Projects_Completed": projects, "Communication_Skill": communication,
        "Backlogs": backlogs, "Extra_Curricular": extra_curricular,
        "Aptitude_Score": aptitude, "Placed": placed
    })


@st.cache_resource
def train_model():
    df = generate_dataset()
    X, y = df[FEATURE_COLS], df["Placed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=RANDOM_SEED
    )
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=False)

    return model, scaler, acc, importance


model, scaler, accuracy, importance = train_model()


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

# Custom neon/gradient styling
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
        color: #111827;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 8px 0 24px 0;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -1px;
        color: #ffffff;
    }

    .hero-title .gradient-text {
        background: linear-gradient(90deg, #d85cff 0%, #9d7bff 42%, #27d9ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        margin-top: 12px;
        color: #c5cbe0;
        font-size: 1.05rem;
    }

    /* Accuracy banner */
    .accuracy-banner {
        background: linear-gradient(100deg, #3d4dff 0%, #237eea 48%, #38c978 100%);
        border-radius: 12px;
        padding: 18px 24px;
        margin: 8px 0 20px 0;
        box-shadow: 0 10px 30px rgba(24, 116, 255, 0.16);
        font-size: 1rem;
        font-weight: 600;
        color: white;
    }

    /* Form card */
    .form-card {
        background: #f8fafc;
        border: 1px solid rgba(120, 165, 230, 0.28);
        border-radius: 16px;
        padding: 24px 26px 8px 26px;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
    }

    /* Streamlit labels and text */
    .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    .stSlider [data-baseweb="slider"] div {
        color: #ffffff;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #c63dff, #4d8dff, #24d5ef) !important;
    }

    .stSlider [role="slider"] {
        background: #55c8ff !important;
        border: 2px solid #ffffff33 !important;
        box-shadow: 0 0 12px rgba(60, 190, 255, 0.65);
    }

    /* Main prediction button */
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        min-height: 54px;
        border: 0 !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1.08rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #7b3cff 0%, #4e7cff 48%, #12cde0 100%) !important;
        box-shadow: 0 10px 28px rgba(81, 99, 255, 0.30);
        transition: transform .18s ease, box-shadow .18s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(49, 189, 255, 0.38);
    }

    /* Result cards */
    .result-card {
        margin-top: 18px;
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(255,255,255,.15);
        background: rgba(15, 28, 54, .88);
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .result-percent {
        font-size: 2.7rem;
        font-weight: 900;
        background: linear-gradient(90deg, #d45cff, #3bdcff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #111827 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aeb6ce;
        font-size: .9rem;
        padding: 26px 0 8px 0;
    }

    /* Mobile */
    @media (max-width: 700px) {
        .hero-title {
            font-size: 2.1rem;
        }
        .hero-subtitle {
            font-size: .95rem;
        }
        .form-card {
            padding: 18px 14px 4px 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 Student <span class="gradient-text">Placement Predictor</span></div>
    <div class="hero-subtitle">Enter your academic &amp; skill details to predict your placement chance.</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="accuracy-banner">
        🏅 &nbsp; Model trained on 1,200 student records • Test accuracy: <b>{accuracy*100:.1f}%</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="form-card">', unsafe_allow_html=True)

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("🎓  CGPA", 0.0, 10.0, 7.5, 0.1)
        iq = st.slider("🧠  IQ", 80, 140, 105)
        internships = st.slider("💼  Internships completed", 0, 3, 1)
        projects = st.slider("📁  Projects completed", 0, 6, 2)

    with col2:
        communication = st.slider("💬  Communication skill (1-10)", 1.0, 10.0, 6.5, 0.1)
        backlogs = st.slider("📚  Active backlogs", 0, 4, 0)
        extra_curricular = st.slider("📖  Extra-curricular score (0-10)", 0.0, 10.0, 5.0, 0.1)
        aptitude = st.slider("🎯  Aptitude test score (0-100)", 0, 100, 65)

    submitted = st.form_submit_button("🚀  Predict Placement", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    student = {
        "CGPA": cgpa, "IQ": iq, "Internships": internships,
        "Projects_Completed": projects, "Communication_Skill": communication,
        "Backlogs": backlogs, "Extra_Curricular": extra_curricular,
        "Aptitude_Score": aptitude
    }

    input_df = pd.DataFrame([student])[FEATURE_COLS]
    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    percentage = round(probability * 100, 2)

    if probability >= 0.5:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">✅ Likely PLACED</div>
                <div class="result-percent">{percentage}%</div>
                <div style="color:#b8c1d9;">Estimated placement probability</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">❌ Likely NOT PLACED</div>
                <div class="result-percent">{percentage}%</div>
                <div style="color:#b8c1d9;">Estimated placement probability</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.progress(int(percentage))

    with st.expander("📊 What influenced this the most?"):
        st.write("Feature importance in the model (overall, not per-student):")
        st.bar_chart(importance)

st.markdown("""
<div class="footer">
    © 2025 Student Placement Predictor &nbsp;|&nbsp; Built with Streamlit 💜
</div>
""", unsafe_allow_html=True)
