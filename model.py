import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# STEP 1 — Load cleaned data
# ─────────────────────────────────────────
print("Step 1: Loading data...")
df = pd.read_csv('cleaned_reviews.csv')

# Drop any rows where clean_review or label is missing
df = df.dropna(subset=['clean_review', 'label'])

X = df['clean_review']   # input: review text
y = df['label']          # output: 0=negative, 1=positive

print(f"  Loaded {len(df)} reviews")

# ─────────────────────────────────────────
# STEP 2 — Split into train and test sets
# ─────────────────────────────────────────
print("Step 2: Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # same split every run
)
print(f"  Training: {len(X_train)} reviews")
print(f"  Testing:  {len(X_test)} reviews")

# ─────────────────────────────────────────
# STEP 3 — Convert text to numbers (TF-IDF)
# ─────────────────────────────────────────
print("Step 3: Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=10000)

X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocab + convert
X_test_tfidf  = vectorizer.transform(X_test)        # convert only

print(f"  Matrix shape: {X_train_tfidf.shape}")

# ─────────────────────────────────────────
# STEP 4 — Train the model
# ─────────────────────────────────────────
print("Step 4: Training model (may take 10-30 seconds)...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
print("  Training complete!")

# ─────────────────────────────────────────
# STEP 5 — Test and measure accuracy
# ─────────────────────────────────────────
print("\nStep 5: Evaluating model...")
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc:.2%}")
print("\n" + classification_report(y_test, y_pred,
      target_names=['negative', 'positive']))

# ─────────────────────────────────────────
# STEP 6 — Save the model and vectorizer
# ─────────────────────────────────────────
print("Step 6: Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("  Saved model.pkl")
print("  Saved vectorizer.pkl")

# ─────────────────────────────────────────
# STEP 7 — Test with your own sentence
# ─────────────────────────────────────────
print("\nStep 7: Testing with a custom sentence...")
import re

stop_words = set([
    'i','me','my','we','our','you','your','he','she','it','they',
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','a','an','the','and','or','but','in','on',
    'at','to','for','of','with','this','that','these','those','so',
    'if','as','by','from','not','no','nor','can','will','just','more'
])

def clean_text(text):
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in stop_words])

# ---- change these sentences and re-run to experiment ----
test_sentences = [
    "This movie was absolutely brilliant, I loved every second!",
    "Terrible film. Waste of time, completely boring and awful.",
    "It was okay, not great but not bad either."
]

print()
for sentence in test_sentences:
    cleaned  = clean_text(sentence)
    vector   = vectorizer.transform([cleaned])
    result   = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0]
    label    = "Positive" if result == 1 else "Negative"
    conf_pct = max(confidence) * 100
    print(f"  Input:  \"{sentence}\"")
    print(f"  Result: {label} ({conf_pct:.0f}% confident)")
    print()

print("Done! Your model is ready.")