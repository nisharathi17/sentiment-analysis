import streamlit as st
import pandas as pd
import pickle
import re

# ── Page config ──
st.set_page_config(page_title="Review Analyser", page_icon="🛒")
st.title("Product Review Sentiment Dashboard")
st.write("Upload a CSV of product reviews to analyse all of them at once.")

# ── Load model (cached) ──
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    vec   = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vec

# ── Text cleaning ──
def clean_text(text):
    stop = set([
        'i','me','my','we','the','a','an','is','was','are','were',
        'and','or','but','in','on','at','to','for','of','with','it'
    ])
    text = str(text)
    text = re.sub(r'<.*?>', '', text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([w for w in text.split() if w not in stop])

# ── Step 1: File upload ──
uploaded_file = st.file_uploader(
    "Upload your reviews CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # Load the CSV
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} reviews!")

    # Show preview
    st.subheader("Preview of your data")
    st.dataframe(df.head(3), use_container_width=True)

    # ── Step 2: Column picker ──
    st.subheader("Select the review column")
    review_col = st.selectbox(
        "Which column contains the review text?",
        options=df.columns.tolist()
    )
    st.write(f"Sample from selected column: *\"{df[review_col].iloc[0]}\"*")

    st.divider()

    # ── Step 3: Analyse button ──
    if st.button("Analyse all reviews", type="primary"):

        model, vectorizer = load_model()

        with st.spinner(f"Analysing {len(df)} reviews..."):

            # Clean all reviews
            df['cleaned'] = df[review_col].apply(clean_text)

            # Vectorize all at once and predict
            vectors          = vectorizer.transform(df['cleaned'])
            preds            = model.predict(vectors)
            probs            = model.predict_proba(vectors)
            df['sentiment']  = ['Positive' if p == 1 else 'Negative' for p in preds]
            df['confidence'] = [round(max(p) * 100) for p in probs]

        st.success("Analysis complete!")
        st.divider()

        # ── Step 4: Dashboard ──
        st.subheader("Summary")

        pos_count = (df['sentiment'] == 'Positive').sum()
        neg_count = (df['sentiment'] == 'Negative').sum()
        avg_conf  = round(df['confidence'].mean())
        pos_pct   = round(pos_count / len(df) * 100)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total reviews",    len(df))
        col2.metric("Positive",         f"{pos_count} ({pos_pct}%)")
        col3.metric("Negative",         neg_count)
        col4.metric("Avg confidence",   f"{avg_conf}%")

        # Bar chart
        st.subheader("Sentiment breakdown")
        st.bar_chart(df['sentiment'].value_counts())

        # Full results table
        st.subheader("Full results")
        st.dataframe(
            df[[review_col, 'sentiment', 'confidence']],
            use_container_width=True
        )

        # ── Step 5: Download button ──
        st.divider()
        csv = df[[review_col, 'sentiment', 'confidence']].to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )