import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string if not a valid text
    text = re.sub(r"\n+", " ", text)  # Replace multiple newlines with a space
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    return text.strip()

# Step 2: Load the datasets
df_current = pd.read_excel("current_affairs.xlsx")
df_non_current = pd.read_excel("non_current_affairs.xlsx")

# Step 3: Identify the correct column dynamically
current_text_col = [col for col in df_current.columns if "sentence" in col.lower()][0]
non_current_text_col = [col for col in df_non_current.columns if "summary" in col.lower()][0]

# Print identified columns for verification
print(f"Identified current affairs text column: {current_text_col}")
print(f"Identified non-current affairs text column: {non_current_text_col}")

# Step 4: Rename columns to a unified name "text"
df_current.rename(columns={current_text_col: "text"}, inplace=True)
df_non_current.rename(columns={non_current_text_col: "text"}, inplace=True)

# Step 5: Clean the text in both datasets
df_current["text"] = df_current["text"].astype(str).apply(clean_text)
df_non_current["text"] = df_non_current["text"].astype(str).apply(clean_text)


# Step 6: Drop rows with missing or empty text
df_current = df_current.dropna(subset=["text"])
df_non_current = df_non_current.dropna(subset=["text"])

# Step 7: Add labels
df_current["label"] = 1
df_non_current["label"] = 0

# Step 8: Balance both datasets
min_len = min(len(df_current), len(df_non_current))
df_current = df_current.sample(min_len, random_state=42)
df_non_current = df_non_current.sample(min_len, random_state=42)

# Step 9: Combine and shuffle
df = pd.concat([df_current, df_non_current], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Step 10: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Step 11: TF-IDF Vectorizer with bigrams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Step 12: Fit and transform
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 13: Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Step 14: Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Step 15: Save the model and vectorizer
joblib.dump(model, "improved_current_affairs_classifier.pkl")
joblib.dump(vectorizer, "improved_vectorizer.pkl")
print("✅ Model and vectorizer saved successfully.")
