"""
Heart Disease Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets users interact with the trained
Support Vector Classifier from the "2 - Heart Disease Prediction" notebook,
without needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook, and
recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.

Visual identity: "Vitals Monitor" -- styled after a bedside patient
monitor (charcoal-green screen, phosphor readouts, monospace numerals),
deliberately distinct from the Titanic app's ledger theme.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Vitals Monitor — Heart Disease Predictor", page_icon="📟", layout="centered")

# ---------------------------------------------------------------------------
# "Vitals Monitor" visual identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

    :root {
        --screen: #10231C;
        --screen-deep: #0A1811;
        --glow: #5FD8A3;
        --amber: #E3A857;
        --cream: #F2EFE6;
        --clinical-red: #C65B4E;
        --panel: #16302A;
    }

    .stApp {
        background: var(--screen);
        color: var(--cream);
        font-family: 'Space Grotesk', sans-serif;
    }

    .monitor-header {
        background: var(--screen-deep);
        margin: -5rem -4rem 2rem -4rem;
        padding: 2.6rem 4rem 2rem 4rem;
        border-bottom: 2px solid var(--glow);
        box-shadow: 0 2px 24px rgba(95, 216, 163, 0.15);
    }
    .monitor-eyebrow {
        color: var(--amber);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        margin: 0 0 0.5rem 0;
    }
    .monitor-title {
        color: var(--cream);
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        line-height: 1.15;
    }
    .monitor-sub {
        color: #B9C9C0;
        font-size: 0.95rem;
        margin-top: 0.7rem;
        max-width: 42rem;
    }
    .monitor-readout {
        display: inline-flex;
        align-items: baseline;
        gap: 0.5rem;
        margin-top: 1.1rem;
        background: var(--panel);
        border: 1px solid var(--glow);
        border-radius: 3px;
        padding: 0.5rem 0.9rem;
    }
    .monitor-readout-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--glow);
    }
    .monitor-readout-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #8FA89C;
    }

    .panel-label {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.05em;
        font-size: 0.85rem;
        color: var(--amber);
        margin-bottom: 0.7rem;
    }
    .panel-rule {
        border: none;
        border-top: 1px dashed #2E4A41;
        margin: 1.5rem 0;
    }

    div[data-baseweb="select"] > div {
        background: var(--panel) !important;
        border: 1px solid #2E4A41 !important;
        color: var(--cream) !important;
        border-radius: 3px;
    }
    label {
        color: #CFE3D9 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
    }

    .stButton > button {
        background: var(--glow);
        color: var(--screen-deep);
        border: none;
        border-radius: 3px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover {
        background: #78E8B5;
    }

    .result-panel {
        border-radius: 3px;
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        border: 1px solid;
    }
    .result-flag {
        background: rgba(198, 91, 78, 0.12);
        border-color: var(--clinical-red);
    }
    .result-clear {
        background: rgba(95, 216, 163, 0.10);
        border-color: var(--glow);
    }
    .result-verdict {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 0.3rem 0;
    }
    .result-flag .result-verdict { color: var(--clinical-red); }
    .result-clear .result-verdict { color: var(--glow); }
    .result-panel p { color: #CFE3D9; margin: 0; }

    .stCaption { color: #7C9488 !important; }
    [data-testid="stExpander"] { border: 1px solid #2E4A41; border-radius: 3px; }
    </style>

    <div class="monitor-header">
        <p class="monitor-eyebrow">BEDSIDE MONITOR · CARDIAC RISK CHANNEL</p>
        <p class="monitor-title">Heart Disease Predictor</p>
        <p class="monitor-sub">
            Enter the four recorded attributes below. A Support Vector
            Classifier, trained on the UCI Cleveland Heart Disease dataset,
            returns a risk reading.
        </p>
        <div class="monitor-readout">
            <span class="monitor-readout-value">0.72</span>
            <span class="monitor-readout-label">TEST ACCURACY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["Sex"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPainType"] = LabelEncoder().fit(
        ["TypicalAngina", "AtypicalAngina", "NonAnginal", "Asymptomatic"]
    )
    encoders["FastingBloodSugar"] = LabelEncoder().fit(["Yes", "No"])
    encoders["ExerciseAngina"] = LabelEncoder().fit(["Yes", "No"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.markdown('<p class="panel-label">PATIENT CHANNEL — INPUT</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["TypicalAngina", "AtypicalAngina", "NonAnginal", "Asymptomatic"],
        index=3,
    )
with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"], index=1)
    exang = st.selectbox("Exercise-Induced Angina?", ["Yes", "No"], index=1)

st.markdown('<hr class="panel-rule">', unsafe_allow_html=True)

if st.button("Read Vitals", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPainType": [cp],
        "FastingBloodSugar": [fbs],
        "ExerciseAngina": [exang],
    })

    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPainType": encoders["ChestPainType"].transform(user_input["ChestPainType"]),
        "FastingBloodSugar": encoders["FastingBloodSugar"].transform(user_input["FastingBloodSugar"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
    })

    prediction = model.predict(encoded)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-panel result-flag">'
            '<p class="result-verdict">⚠ HEART DISEASE PREDICTED</p>'
            '<p>The model flags this reading as consistent with heart disease.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-panel result-clear">'
            '<p class="result-verdict">✓ NO HEART DISEASE PREDICTED</p>'
            '<p>The model does not flag this reading as consistent with heart disease.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.markdown('<hr class="panel-rule">', unsafe_allow_html=True)
st.caption(
    "Built for teaching purposes (AIC354 Assignment 2) from a 303-row subset "
    "of the UCI Cleveland Heart Disease dataset, restricted to 4 input "
    "attributes. Predictions reflect patterns in this small dataset only, "
    "not a rigorous clinical analysis -- do not use for real medical decisions."
)
