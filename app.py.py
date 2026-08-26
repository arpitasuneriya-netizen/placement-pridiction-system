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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "student_id" not in st.session_state:
    st.session_state.student_id = ""



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

# Custom clean white styling
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#ffffff,#f4f7fb); color:#1e293b; }
.block-container { max-width:1120px; padding-top:2.2rem; padding-bottom:2rem; }
.hero { text-align:center; padding:8px 0 24px; }
.hero-title { font-size:3rem; font-weight:800; color:#172554; }
.hero-title .gradient-text { background:linear-gradient(90deg,#2563eb,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-subtitle { color:#64748b; font-size:1.05rem; }
.accuracy-banner { background:linear-gradient(100deg,#2563eb,#4f46e5,#7c3aed); border-radius:12px; padding:18px 24px; margin:8px 0 20px; color:white; font-weight:600; }
.form-card,.login-card,.result-card { background:#ffffff; border:1px solid #dbe4f0; border-radius:16px; padding:24px; box-shadow:0 12px 35px rgba(15,23,42,.10); }
label,.stNumberInput label,.stTextInput label { color:#334155 !important; font-weight:700 !important; }
div.stButton>button,div[data-testid="stFormSubmitButton"]>button { width:100%; min-height:52px; border:0 !important; border-radius:12px !important; color:white !important; font-size:1.05rem !important; font-weight:800 !important; background:linear-gradient(90deg,#2563eb,#4f46e5,#7c3aed) !important; }
.result-title { font-size:1.45rem; font-weight:800; color:#1e293b; }
.result-percent { font-size:2.7rem; font-weight:900; color:#4f46e5; }
.footer { text-align:center; color:#64748b; padding:26px 0 8px; }
</style>
""", unsafe_allow_html=True)

# Login page
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🎓 Student <span class="gradient-text">Placement Predictor</span></div>
      <div class="hero-subtitle">Login with the details of a particular student.</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.subheader("🔐 Student Login")
    with st.form("login_form"):
        name = st.text_input("👤 Student Name", placeholder="Enter student name")
        sid = st.text_input("🆔 Enrollment / Student ID", placeholder="Enter student ID")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
        login = st.form_submit_button("Login")
    st.markdown('</div>', unsafe_allow_html=True)
    if login:
        if name.strip() and sid.strip() and password.strip():
            st.session_state.logged_in = True
            st.session_state.student_name = name.strip()
            st.session_state.student_id = sid.strip()
            st.rerun()
        else:
            st.error("Please fill in all login details.")
    st.stop()



st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 Student <span class="gradient-text">Placement Predictor</span></div>
    <div class="hero-subtitle">Enter your academic &amp; skill details to predict your placement chance.</div>
</div>
""", unsafe_allow_html=True)

st.info(f"👤 Logged in as: **{st.session_state.student_name}**  |  🆔 Student ID: **{st.session_state.student_id}**")
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

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
        cgpa = st.number_input("🎓  CGPA", value=None, placeholder="Type any value", format="%.2f")
        iq = st.number_input("🧠  IQ", value=None, placeholder="Type any value", format="%.2f")
        internships = st.number_input("💼  Internships completed", value=None, placeholder="Type any value", format="%.2f")
        projects = st.number_input("📁  Projects completed", value=None, placeholder="Type any value", format="%.2f")

    with col2:
        communication = st.number_input("💬  Communication skill", value=None, placeholder="Type any value", format="%.2f")
        backlogs = st.number_input("📚  Active backlogs", value=None, placeholder="Type any value", format="%.2f")
        extra_curricular = st.number_input("📖  Extra-curricular score", value=None, placeholder="Type any value", format="%.2f")
        aptitude = st.number_input("🎯  Aptitude test score", value=None, placeholder="Type any value", format="%.2f")

    submitted = st.form_submit_button("🚀  Predict Placement", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    if any(v is None for v in [cgpa, iq, internships, projects, communication, backlogs, extra_curricular, aptitude]):
        st.error("Please enter all values before predicting.")
        st.stop()

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
