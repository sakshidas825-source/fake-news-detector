import streamlit as st
import joblib
import re

# ---------- Page config ----------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------- Load model + vectorizer ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------- Text cleaning (must match training) ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "Real" if pred == 1 else "Fake"
    confidence = proba[pred] * 100
    return label, confidence

# ---------- Header ----------
st.title("📰 Fake News Detector")
st.markdown(
    "An AI model that classifies news headlines/articles as **Fake** or **Real**, "
    "trained on a labeled dataset of ~45,000 news articles using TF-IDF + Logistic Regression."
)

st.divider()

# ---------- Model stats ----------
col1, col2, col3 = st.columns(3)
col1.metric("Model Accuracy", "98.7%")
col2.metric("Training Articles", "44,898")
col3.metric("Algorithm", "TF-IDF + LogReg")

st.divider()

# ---------- Example headlines ----------
st.subheader("Try an example")
examples = {
    "Select an example...": "",
    "Likely Fake example": "Scientists confirm the moon is actually made of cheese, NASA reveals in shocking new report",
    "Likely Real example": "The Federal Reserve raised interest rates by a quarter point on Wednesday, citing continued inflation concerns.",
}
choice = st.selectbox("Quick test headlines:", list(examples.keys()))

# ---------- Text input ----------
default_text = examples[choice] if choice != "Select an example..." else ""
user_input = st.text_area(
    "Or paste your own headline / article text:",
    value=default_text,
    height=150,
    placeholder="Paste a news headline or article text here..."
)

# ---------- Predict ----------
if st.button("Check News", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text or select an example above.")
    else:
        label, confidence = predict_news(user_input)

        if label == "Real":
            st.success(f"✅ Prediction: **REAL** ({confidence:.1f}% confidence)")
        else:
            st.error(f"🚫 Prediction: **FAKE** ({confidence:.1f}% confidence)")

        st.progress(confidence / 100)

st.divider()

# ---------- How it works ----------
with st.expander("ℹ️ How this works"):
    st.markdown(
        """
        1. **Dataset:** ~45,000 labeled news articles (Fake and Real) sourced from a public Kaggle dataset.
        2. **Text cleaning:** lowercasing, URL removal, punctuation stripping.
        3. **Vectorization:** TF-IDF converts text into numerical features, weighting words
           by how distinctive they are to a document (not just frequency).
        4. **Classification:** a Logistic Regression model trained on these features predicts
           Fake vs Real, along with a confidence score.
        """
    )

with st.expander("⚠️ Limitations"):
    st.markdown(
        """
        - This model achieves high accuracy **on this specific dataset**, partly because the
          Fake and Real articles in it come from fairly distinct sources with different writing
          styles — this makes the task easier than real-world fake news detection.
        - It does not fact-check claims, verify sources, or understand context the way a human
          fact-checker would — it's a statistical text pattern classifier, not a truth engine.
        - Performance on genuinely novel, out-of-distribution news (e.g. very recent events,
          different publication styles) would likely be lower than the 98.7% shown here.
        - A production system would need continuous retraining, source verification, and
          ideally a human-in-the-loop review step for high-stakes classifications.
        """
    )

st.caption("Built with Streamlit · scikit-learn · TF-IDF + Logistic Regression")
