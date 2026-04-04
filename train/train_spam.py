import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("data/spam.csv")

# Rename columns
df = df.rename(columns={"sms": "text", "label": "label"})

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

# Train
model = MultinomialNB()
model.fit(X, y)

# Save
pickle.dump(model, open("models/spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/spam_vectorizer.pkl", "wb"))

print("Spam model trained!")