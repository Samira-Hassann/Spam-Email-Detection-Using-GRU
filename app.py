import streamlit as st
import tensorflow as tf
import joblib
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="MailGuard - Spam Detector", page_icon="📧", layout="centered")

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('gru_spam_model.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Error loading model files: {e}")

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("📧 MailGuard: Spam Detection System")
st.write("Enter an email or message below to check whether it's **Spam** or **Ham** using our GRU Model.")

user_input = st.text_area("Message Content:", placeholder="Paste your email text here...", height=150)

if st.button("Classify Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        cleaned_input = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned_input])
        padded_seq = pad_sequences(seq, maxlen=250, padding='post', truncating='post')
        
        prediction_prob = model.predict(padded_seq)[0][0]
        
        threshold = 0.16 
        
        st.divider()
        if prediction_prob >= threshold:
            st.error(f"🚨 **SPAM DETECTED** (Probability: {prediction_prob:.2%})")
        else:
            st.success(f"✅ **HAM (Legitimate Message)** (Spam Probability: {prediction_prob:.2%})")
