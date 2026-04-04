import streamlit as st
import pickle
import re

# Load models
sentiment_model = pickle.load(open("models/sentiment_model.pkl", "rb"))
sentiment_vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

spam_model = pickle.load(open("models/spam_model.pkl", "rb"))
spam_vectorizer = pickle.load(open("models/spam_vectorizer.pkl", "rb"))

emotion_model = pickle.load(open("models/emotion_model.pkl", "rb"))
emotion_vectorizer = pickle.load(open("models/emotion_vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Emotion labels
emotion_map = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

# UI
st.set_page_config(page_title="AI Text Analyzer", layout="centered")

st.title("🧠 AI Text Analyzer Suite")
st.write("Analyze text using multiple AI models")

task = st.selectbox("Choose Task", [
    "Sentiment Analysis",
    "Spam Detection",
    "Emotion Detection"
])

text = st.text_area("Enter your text:")

if st.button("Analyze"):

    cleaned = clean_text(text)

    if task == "Sentiment Analysis":
        vec = sentiment_vectorizer.transform([cleaned])
        pred = sentiment_model.predict(vec)[0]
        prob = sentiment_model.predict_proba(vec)[0]

        if pred == 1:
            st.success(f"Positive 😊 ({max(prob)*100:.2f}%)")
        else:
            st.error(f"Negative 😡 ({max(prob)*100:.2f}%)")

    elif task == "Spam Detection":
        vec = spam_vectorizer.transform([cleaned])
        pred = spam_model.predict(vec)[0]

        if pred == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")

    elif task == "Emotion Detection":
        vec = emotion_vectorizer.transform([cleaned])
        pred = emotion_model.predict(vec)[0]

        st.info(f"Emotion: {emotion_map[pred]}")