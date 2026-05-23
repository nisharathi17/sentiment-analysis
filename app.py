import streamlit as st
import pickle
import re
from transformers import pipeline

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎬",
    layout="centered"
)

# ─────────────────────────────────────────
# Load both models (cached — loads once)
# ─────────────────────────────────────────
@st.cache_resource
def load_lr():
    model = pickle.load(open('model.pkl', 'rb'))
    vec   = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vec

@st.cache_resource
def load_bert():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# ─────────────────────────────────────────
# Text cleaning (for LR model only)
# ─────────────────────────────────────────
def clean_text(text):
    stop_words = set([
        'i','me','my','we','our','you','your','he','she','it','they',
        'is','are','was','were','be','been','being','have','has','had',
        'do','does','did','a','an','the','and','or','but','in','on',
        'at','to','for','of','with','this','that','these','those','so',
        'if','as','by','from','not','no','nor','can','will','just','more'
    ])
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in stop_words])

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("Sentiment Analyser")
st.write("Type a movie review and find out if it's positive or negative.")

st.divider()

# Model selector
model_choice = st.radio(
    "Choose your model",
    ["Logistic Regression (fast)", "DistilBERT (accurate)"],
    horizontal=True
)

# Show a tip depending on model chosen
if model_choice == "Logistic Regression (fast)":
    st.info("Fast and lightweight. Great for clear positive/negative reviews. May struggle with negation like 'not bad'.")
else:
    st.info("Context-aware deep learning model. Understands negation, sarcasm hints, and complex sentences.")

st.divider()

# Text input
user_input = st.text_area(
    "Your review",
    placeholder="e.g. This movie was absolutely brilliant, I loved every second!",
    height=150
)

# Analyse button
if st.button("Analyse", type="primary"):
    if user_input.strip():

        with st.spinner("Analysing..."):

            # ── Logistic Regression path ──
            if model_choice == "Logistic Regression (fast)":
                lr_model, vectorizer = load_lr()
                cleaned    = clean_text(user_input)
                vector     = vectorizer.transform([cleaned])
                result     = lr_model.predict(vector)[0]
                confidence = lr_model.predict_proba(vector)[0]
                conf_pct   = round(max(confidence) * 100)
                label      = "Positive" if result == 1 else "Negative"
                is_positive = result == 1

                # Show cleaned text in expander
                with st.expander("See cleaned text (what the model read)"):
                    st.write(cleaned)

            # ── DistilBERT path ──
            else:
                bert       = load_bert()
                result     = bert(user_input[:512])[0]
                label      = result['label'].capitalize()
                conf_pct   = round(result['score'] * 100)
                is_positive = result['label'] == 'POSITIVE'

        # ── Show result ──
        st.divider()

        if is_positive:
            st.success(f"Positive review ({conf_pct}% confident)")
        else:
            st.error(f"Negative review ({conf_pct}% confident)")

        # Confidence metric
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{conf_pct}%")
        with col2:
            st.metric("Model used", "LR" if "Logistic" in model_choice else "DistilBERT")

        # Confidence interpretation
        if conf_pct >= 90:
            st.caption("Very confident — clear signal in the review.")
        elif conf_pct >= 75:
            st.caption("Moderately confident — some ambiguity in the text.")
        else:
            st.caption("Low confidence — the review may be mixed or ambiguous.")

    else:
        st.warning("Please type a review first!")

# ─────────────────────────────────────────
# Try both models section
# ─────────────────────────────────────────
st.divider()
with st.expander("Compare both models on the same review"):
    compare_input = st.text_area(
        "Type a review to compare",
        placeholder="e.g. This movie was not bad at all",
        height=100,
        key="compare"
    )
    if st.button("Run both models"):
        if compare_input.strip():
            with st.spinner("Running both models..."):
                # LR prediction
                lr_model, vectorizer = load_lr()
                cleaned  = clean_text(compare_input)
                vector   = vectorizer.transform([cleaned])
                lr_pred  = lr_model.predict(vector)[0]
                lr_conf  = round(max(lr_model.predict_proba(vector)[0]) * 100)
                lr_label = "Positive" if lr_pred == 1 else "Negative"

                # BERT prediction
                bert      = load_bert()
                bert_res  = bert(compare_input[:512])[0]
                bert_label = bert_res['label'].capitalize()
                bert_conf  = round(bert_res['score'] * 100)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Logistic Regression")
                if lr_pred == 1:
                    st.success(f"{lr_label} ({lr_conf}%)")
                else:
                    st.error(f"{lr_label} ({lr_conf}%)")

            with col2:
                st.subheader("DistilBERT")
                if bert_res['label'] == 'POSITIVE':
                    st.success(f"{bert_label} ({bert_conf}%)")
                else:
                    st.error(f"{bert_label} ({bert_conf}%)")

            if lr_label.lower() != bert_label.lower():
                st.warning("The models disagree! This usually happens with negation, sarcasm, or ambiguous phrasing — exactly where DistilBERT shines.")
            else:
                st.info("Both models agree on this one.")
        else:
            st.warning("Please type a review to compare!")