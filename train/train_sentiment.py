import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Loading big dataset...")

df = pd.read_csv(
    "data/sentiment_big.csv",
    encoding="latin-1",
    header=None
)

# Shuffle dataset
df = df.sample(n=200000, random_state=42)

df.columns = ["target", "id", "date", "flag", "user", "text"]

print("Initial shape:", df.shape)

# Keep only needed
df = df[["text", "target"]]

# Map labels
df["target"] = df["target"].map({0: 0, 4: 1})

# Drop invalid
df = df.dropna()

print("After mapping:", df.shape)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned"] = df["text"].apply(clean_text)

# Remove empty
df = df[df["cleaned"].str.strip() != ""]

print("After cleaning:", df.shape)

# 🔥 IMPORTANT: CHECK BEFORE BALANCING
df_pos = df[df["target"] == 1]
df_neg = df[df["target"] == 0]

print("Positive samples:", len(df_pos))
print("Negative samples:", len(df_neg))

# If empty → STOP EARLY
if len(df_pos) == 0 or len(df_neg) == 0:
    print("❌ ERROR: One class is empty. Check dataset.")
    exit()

# Safe balancing
min_size = min(len(df_pos), len(df_neg), 50000)

df_pos = df_pos.sample(min_size, random_state=42)
df_neg = df_neg.sample(min_size, random_state=42)

df = pd.concat([df_pos, df_neg])

print("Balanced dataset:", df.shape)

# Vectorize
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["cleaned"])
y = df["target"]

print("Training model...")

model = LogisticRegression(max_iter=300)
model.fit(X, y)

# Save
pickle.dump(model, open("models/sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("🔥 FINAL Sentiment model trained!")