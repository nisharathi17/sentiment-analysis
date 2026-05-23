import streamlit as st
import pickle, re
 
# Load once and cache in memory
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    vec   = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vec
 
model, vectorizer = load_model()
 
st.title("Sentiment Analyser")
st.write("Type a movie review and find out if it's positive or negative.")
 
# Multi-line text input box
user_input = st.text_area(
    "Your review",
    placeholder="e.g. This movie was absolutely brilliant...",
    height=150
)
 
def clean_text(text):
    stop = set(['i','me','the','a','an','is','was',
                'and','or','in','it','of','to'])
    text = re.sub(r'<.*?>','',text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]',' ',text)   # fixed: \s not s
    text = re.sub(r'\s+',' ',text).strip()        # fixed: \s+ not s+
    return ' '.join([w for w in text.split() if w not in stop])
 
if st.button("Analyse"):
    if user_input:                                # fixed: indentation
        cleaned    = clean_text(user_input)
        vector     = vectorizer.transform([cleaned])
        result     = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]
        conf_pct   = round(max(confidence) * 100)
 
        # Show result with colour
        if result == 1:
            st.success(f"Positive review ({conf_pct}% confident)")
        else:
            st.error(f"Negative review ({conf_pct}% confident)")
 
        # Show confidence as a metric
        st.metric("Confidence", f"{conf_pct}%")
 
        # Show what the cleaned text looked like
        with st.expander("See cleaned text"):
            st.write(cleaned)
 
    else:
        st.warning("Please type a review first!")