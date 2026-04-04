import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/emotion.csv")

# Keep required columns
df = df[["text", "label"]]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["cleaned"] = df["text"].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save
pickle.dump(model, open("models/emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/emotion_vectorizer.pkl", "wb"))

print("Emotion model trained!")