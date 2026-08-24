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
st.title("🎓 Student Placement Predictor")
st.caption("Enter your academic & skill details to predict your placement chance.")
st.info(f"Model trained on 1,200 student records • Test accuracy: **{accuracy*100:.1f}%**")

with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        iq = st.slider("IQ", 80, 140, 105)
        internships = st.slider("Internships completed", 0, 3, 1)
        projects = st.slider("Projects completed", 0, 6, 2)

    with col2:
        communication = st.slider("Communication skill (1-10)", 1.0, 10.0, 6.5, 0.1)
        backlogs = st.slider("Active backlogs", 0, 4, 0)
        extra_curricular = st.slider("Extra-curricular score (0-10)", 0.0, 10.0, 5.0, 0.1)
        aptitude = st.slider("Aptitude test score (0-100)", 0, 100, 65)

    submitted = st.form_submit_button("Predict Placement", use_container_width=True)

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

    st.divider()
    if probability >= 0.5:
        st.success(f"### ✅ Likely PLACED — {percentage}% chance")
    else:
        st.error(f"### ❌ Likely NOT PLACED — {percentage}% chance")

    st.progress(int(percentage))

    with st.expander("What influenced this the most?"):
        st.write("Feature importance in the model (overall, not per-student):")
        st.bar_chart(importance)

st.divider()
st.caption("Built with Streamlit • Random Forest classifier • Synthetic training data (swap in a real dataset for production use)")
