import streamlit as st
import pickle
import sys
import os

# -------------------------------
# Page config (must be first)
# -------------------------------
st.set_page_config(page_title="Spam SMS Detector", page_icon="📱")

# -------------------------------
# Paths & imports
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from preprocess import clean_text

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE_DIR, "model", "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------
# Session state init
# -------------------------------
if "message" not in st.session_state:
    st.session_state.message = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "prob" not in st.session_state:
    st.session_state.prob = None

if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

# -------------------------------
# Handle clear safely BEFORE widget
# -------------------------------
if st.session_state.clear_flag:
    st.session_state.message = ""
    st.session_state.result = None
    st.session_state.prob = None
    st.session_state.clear_flag = False

# -------------------------------
# UI
# -------------------------------
st.title("📱 Spam SMS Detector")
st.write("Detect whether a message is **Spam** or **Legitimate** using NLP + ML")

# Input
message = st.text_area(
    "Enter your message",
    height=150,
    placeholder="Type SMS here...",
    key="message"
)

# Buttons (correct position)
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    predict_clicked = st.button("Predict", key="predict_btn")

with col3:
    clear_clicked = st.button("Clear", key="clear_btn")

# -------------------------------
# Clear trigger
# -------------------------------
if clear_clicked:
    st.session_state.clear_flag = True
    st.rerun()

# -------------------------------
# Prediction
# -------------------------------
if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vector)[0][1]

        st.session_state.result = result
        st.session_state.prob = prob

# -------------------------------
# Display result
# -------------------------------
if st.session_state.result is not None:
    if st.session_state.result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Legitimate Message")

    if st.session_state.prob is not None:
        st.caption(f"Spam probability: {st.session_state.prob:.2f}")