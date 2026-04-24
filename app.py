import streamlit as st
import pickle
import sys
import os

# Ensure src is importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from preprocess import clean_text

# Load model + vectorizer safely
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "model", "model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_artifacts()

# Page config (must be first Streamlit command)
st.set_page_config(page_title="Spam SMS Detector", page_icon="📱")

# UI
st.title("📱 Spam SMS Detector")
st.write("Detect whether a message is **Spam** or **Legitimate** using NLP + ML")

# Input
message = st.text_area("Enter your message", height=150, placeholder="Type SMS here...")

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("Predict")

with col2:
    clear_clicked = st.button("Clear")

if clear_clicked:
    st.rerun()

# Prediction
if predict_clicked:
    if not message.strip():
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        # If your model supports probabilities (e.g., LogisticRegression)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vector)[0][1]

        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Legitimate Message")

        if prob is not None:
            st.caption(f"Spam probability: {prob:.2f}")
