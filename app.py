import streamlit as st
import pickle
import re
import pandas as pd
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
# Sidebar navigation
# ─────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a tool",
    ["Single Review Analyser", "Bulk Product Review Dashboard"]
)
st.sidebar.divider()
st.sidebar.caption("Built with Logistic Regression + DistilBERT")

# ─────────────────────────────────────────
# Shared: load models (cached)
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
# Shared: text cleaning
# ─────────────────────────────────────────
def clean_text(text):
    stop_words = set([
        'i','me','my','we','our','you','your','he','she','it','they',
        'is','are','was','were','be','been','being','have','has','had',
        'do','does','did','a','an','the','and','or','but','in','on',
        'at','to','for','of','with','this','that','these','those','so',
        'if','as','by','from','not','no','nor','can','will','just','more'
    ])
    text = str(text)
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in stop_words])

# ═════════════════════════════════════════
# PAGE 1 — Single Review Analyser
# ═════════════════════════════════════════
if page == "Single Review Analyser":

    st.title("Sentiment Analyser")
    st.write("Type a movie review and find out if it's positive or negative.")
    st.divider()

    # Model selector
    model_choice = st.radio(
        "Choose your model",
        ["Logistic Regression (fast)", "DistilBERT (accurate)"],
        horizontal=True
    )

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

    if st.button("Analyse", type="primary"):
        if user_input.strip():
            with st.spinner("Analysing..."):

                if model_choice == "Logistic Regression (fast)":
                    lr_model, vectorizer = load_lr()
                    cleaned    = clean_text(user_input)
                    vector     = vectorizer.transform([cleaned])
                    result     = lr_model.predict(vector)[0]
                    confidence = lr_model.predict_proba(vector)[0]
                    conf_pct   = round(max(confidence) * 100)
                    is_positive = result == 1
                    with st.expander("See cleaned text"):
                        st.write(cleaned)

                else:
                    bert        = load_bert()
                    result      = bert(user_input[:512])[0]
                    conf_pct    = round(result['score'] * 100)
                    is_positive = result['label'] == 'POSITIVE'

            st.divider()
            if is_positive:
                st.success(f"Positive review ({conf_pct}% confident)")
            else:
                st.error(f"Negative review ({conf_pct}% confident)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{conf_pct}%")
            with col2:
                st.metric("Model used", "LR" if "Logistic" in model_choice else "DistilBERT")

            if conf_pct >= 90:
                st.caption("Very confident — clear signal in the review.")
            elif conf_pct >= 75:
                st.caption("Moderately confident — some ambiguity in the text.")
            else:
                st.caption("Low confidence — the review may be mixed or ambiguous.")

        else:
            st.warning("Please type a review first!")

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
                    lr_model, vectorizer = load_lr()
                    cleaned  = clean_text(compare_input)
                    vector   = vectorizer.transform([cleaned])
                    lr_pred  = lr_model.predict(vector)[0]
                    lr_conf  = round(max(lr_model.predict_proba(vector)[0]) * 100)
                    lr_label = "Positive" if lr_pred == 1 else "Negative"

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
                    st.warning("The models disagree! This usually happens with negation or complex phrasing — exactly where DistilBERT shines.")
                else:
                    st.info("Both models agree on this one.")
            else:
                st.warning("Please type a review to compare!")

# ═════════════════════════════════════════
# PAGE 2 — Bulk Product Review Dashboard
# ═════════════════════════════════════════
elif page == "Bulk Product Review Dashboard":

    st.title("Product Review Dashboard")
    st.write("Upload a CSV of product reviews to analyse all of them at once.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your reviews CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} reviews!")

        st.subheader("Preview")
        st.dataframe(df.head(3), use_container_width=True)

        st.divider()
        review_col = st.selectbox(
            "Which column contains the review text?",
            options=df.columns.tolist()
        )
        st.write(f"Sample: *\"{str(df[review_col].iloc[0])[:100]}...\"*")

        st.divider()
        if st.button("Analyse all reviews", type="primary"):
            lr_model, vectorizer = load_lr()

            with st.spinner(f"Analysing {len(df)} reviews..."):
                df['cleaned']    = df[review_col].apply(clean_text)
                vectors          = vectorizer.transform(df['cleaned'])
                preds            = lr_model.predict(vectors)
                probs            = lr_model.predict_proba(vectors)
                df['sentiment']  = ['Positive' if p == 1 else 'Negative' for p in preds]
                df['confidence'] = [round(max(p) * 100) for p in probs]

            st.success("Analysis complete!")
            st.divider()

            # Metrics
            pos_count = (df['sentiment'] == 'Positive').sum()
            neg_count = (df['sentiment'] == 'Negative').sum()
            avg_conf  = round(df['confidence'].mean())
            pos_pct   = round(pos_count / len(df) * 100)

            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total reviews",  len(df))
            col2.metric("Positive",       f"{pos_count} ({pos_pct}%)")
            col3.metric("Negative",       neg_count)
            col4.metric("Avg confidence", f"{avg_conf}%")

            # Chart
            st.subheader("Sentiment breakdown")
            st.bar_chart(df['sentiment'].value_counts())

            # Full results table
            st.subheader("Full results")
            st.dataframe(
                df[[review_col, 'sentiment', 'confidence']],
                use_container_width=True
            )

            # Download
            st.divider()
            csv = df[[review_col, 'sentiment', 'confidence']].to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )