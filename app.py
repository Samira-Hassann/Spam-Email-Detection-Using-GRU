import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# Configuration
# =========================

MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

MAX_LEN = 250
THRESHOLD = 0.16


# =========================
# Load Model & Tokenizer
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as file:
        return pickle.load(file)


model = load_model()
tokenizer = load_tokenizer()


# =========================
# Text Preprocessing
# =========================

def preprocess_text(text):
    """
    Apply the same basic preprocessing used during training.
    """

    text = text.lower()

    # Tokenize using the same tokenizer used during training
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence to the same length used during training
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded_sequence


# =========================
# Prediction
# =========================

def predict_spam(text):

    processed_text = preprocess_text(text)

    probability = float(
        model.predict(processed_text, verbose=0)[0][0]
    )

    prediction = int(probability >= THRESHOLD)

    return prediction, probability


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam Message Detector")

st.write(
    "Enter a message and the model will classify it as "
    "**Ham** or **Spam**."
)

message = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Type your message here..."
)

if st.button("🔍 Analyze Message", use_container_width=True):

    if not message.strip():

        st.warning("⚠️ Please enter a message first.")

    else:

        prediction, probability = predict_spam(message)

        st.divider()

        if prediction == 1:

            st.error("🚨 SPAM")

            st.metric(
                "Spam Probability",
                f"{probability:.2%}"
            )

        else:

            st.success("✅ HAM")

            st.metric(
                "Spam Probability",
                f"{probability:.2%}"
            )

        st.caption(
            f"Classification threshold: {THRESHOLD}"
        )
