"""
Titanic Passenger Survival Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets users interact with the trained
Support Vector Classifier from the Titanic notebook, without needing to
install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook, and
recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.

Visual identity: "RMS Manifest" -- styled after a period passenger
ledger (deep navy, brass rule lines, parchment paper), rather than the
default Streamlit look.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="RMS Manifest — Survival Predictor", page_icon="⚓", layout="centered")

# ---------------------------------------------------------------------------
# "RMS Manifest" visual identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@400;600&display=swap');

    :root {
        --navy: #0B2545;
        --navy-deep: #081b36;
        --parchment: #F4EDE1;
        --parchment-dim: #EAE0CC;
        --brass: #C9A227;
        --ink: #241F14;
        --teal: #2E6F6E;
        --brick: #8B3A3A;
    }

    .stApp {
        background: var(--parchment);
        color: var(--ink);
        font-family: 'Source Sans 3', sans-serif;
    }

    /* Header band */
    .manifest-header {
        background: linear-gradient(180deg, var(--navy) 0%, var(--navy-deep) 100%);
        margin: -5rem -4rem 2rem -4rem;
        padding: 2.8rem 4rem 2rem 4rem;
        border-bottom: 3px solid var(--brass);
    }
    .manifest-eyebrow {
        color: var(--brass);
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        margin: 0 0 0.4rem 0;
    }
    .manifest-title {
        color: var(--parchment);
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 2.3rem;
        margin: 0;
        line-height: 1.15;
    }
    .manifest-sub {
        color: #C7D2DE;
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.95rem;
        margin-top: 0.6rem;
        max-width: 42rem;
    }
    .manifest-accuracy {
        display: inline-block;
        margin-top: 1rem;
        color: var(--navy);
        background: var(--brass);
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        border-radius: 2px;
    }

    /* Section labels styled like ledger column headings */
    .ledger-rule {
        border: none;
        border-top: 1px solid var(--brass);
        margin: 1.6rem 0 1.2rem 0;
    }
    .ledger-label {
        font-family: 'Playfair Display', serif;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--navy);
        margin-bottom: 0.6rem;
    }

    /* Inputs */
    div[data-baseweb="select"] > div {
        background: #FFFEFB;
        border: 1px solid #C9BFA6 !important;
        border-radius: 2px;
    }
    label {
        color: var(--ink) !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Button */
    .stButton > button {
        background: var(--navy);
        color: var(--parchment);
        border: 1px solid var(--brass);
        border-radius: 2px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover {
        background: var(--navy-deep);
        border-color: var(--brass);
        color: var(--brass);
    }

    /* Result panel */
    .result-panel {
        border: 1px solid var(--brass);
        background: #FFFEFB;
        padding: 1.4rem 1.6rem;
        border-radius: 2px;
        margin-top: 1rem;
    }
    .result-survived { border-left: 6px solid var(--teal); }
    .result-lost { border-left: 6px solid var(--brick); }
    .result-verdict {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    .result-survived .result-verdict { color: var(--teal); }
    .result-lost .result-verdict { color: var(--brick); }

    .stCaption, .st-emotion-cache-1629p8f, p.manifest-footer {
        color: #5B5440 !important;
    }
    </style>

    <div class="manifest-header">
        <p class="manifest-eyebrow">RMS TITANIC · PASSENGER MANIFEST, DIGITISED</p>
        <p class="manifest-title">Survival Prediction Ledger</p>
        <p class="manifest-sub">
            Enter a passenger's recorded class, gender, sibling/spouse count,
            and port of embarkation. A Support Vector Classifier, trained on
            the historical passenger record, returns a survival prediction.
        </p>
        <span class="manifest-accuracy">Model accuracy: 80% on held-out records</span>
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
    encoders["PClass"] = LabelEncoder().fit(["First", "Second", "Third"])
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["Sibling"] = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
    encoders["Embarked"] = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.markdown('<p class="ledger-label">Passenger Particulars</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"], index=2)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    sibling = st.selectbox("Siblings / Spouses Aboard", ["Zero", "One", "Two", "Three"])
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)

if st.button("Consult the Manifest", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "PClass": [pclass],
        "Gender": [gender],
        "Sibling": [sibling],
        "Embarked": [embarked],
    })

    encoded = pd.DataFrame({
        "PClass": encoders["PClass"].transform(user_input["PClass"]),
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "Sibling": encoders["Sibling"].transform(user_input["Sibling"]),
        "Embarked": encoders["Embarked"].transform(user_input["Embarked"]),
    })

    prediction = model.predict(encoded)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-panel result-survived">'
            '<p class="result-verdict">Survived</p>'
            '<p>The model places this passenger among those recorded as survivors.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-panel result-lost">'
            '<p class="result-verdict">Did Not Survive</p>'
            '<p>The model places this passenger among those recorded as lost.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.markdown('<hr class="ledger-rule">', unsafe_allow_html=True)
st.caption(
    "Built for teaching purposes from a 100-row simplified Titanic dataset. "
    "Predictions reflect patterns in this small toy dataset, not a rigorous "
    "historical analysis."
)
