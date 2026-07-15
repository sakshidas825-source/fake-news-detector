# 📰 Fake News Detector

An AI-powered web app that classifies news headlines and articles as **Fake** or **Real**, built with a TF-IDF + Logistic Regression pipeline and deployed as a live Streamlit app.

**Live demo:** https://fake-news-detector-dawpnwdmyhwncud8szgsks.streamlit.app

---

## Overview

This project trains a text classification model to detect fake news using a labeled dataset of ~45,000 real-world news articles. Rather than relying on deep learning or an external LLM API, it uses a classical, explainable NLP approach — TF-IDF vectorization paired with Logistic Regression — which is fast to train, easy to reason about, and still highly effective on this task.

## How It Works

1. **Dataset:** [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle) — 23,481 fake articles and 21,417 real articles, 44,898 total.
2. **Text cleaning:** lowercasing, URL removal, punctuation/number stripping, whitespace normalization.
3. **Vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency) converts article text into numerical features, weighting words by how distinctive they are to a document rather than just how often they appear.
4. **Classification:** a Logistic Regression model is trained on the vectorized text to predict Fake (0) or Real (1), along with a confidence score.
5. **Deployment:** the trained model and vectorizer are saved with `joblib` and loaded into a Streamlit app for live, interactive predictions.

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.7% |
| Precision (Fake) | 0.99 |
| Recall (Fake) | 0.98 |
| Precision (Real) | 0.98 |
| Recall (Real) | 0.99 |

Train/test split: 80/20 (35,918 train / 8,980 test articles), stratified by label.

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (TF-IDF, Logistic Regression)
- **Model persistence:** joblib
- **Web app:** Streamlit
- **Training environment:** Google Colab
- **Deployment:** Streamlit Community Cloud

## Project Structure

```
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── fake_news_model.pkl       # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
#screenshot: <img width="959" height="539" alt="Screenshot 2026-07-15 141803" src="https://github.com/user-attachments/assets/3269ca75-c4e0-4378-a576-5914271839ce" />
<img width="951" height="539" alt="Screenshot 2026-07-15 141748" src="https://github.com/user-attachments/assets/d0e441c8-a23e-4ca0-baaf-c3f0e700e01d" />


## Limitations

- This model achieves high accuracy **on this specific dataset**, partly because the Fake and Real articles in it come from fairly distinct sources with different writing styles — this makes the classification task easier than real-world fake news detection.
- It does not fact-check claims, verify sources, or understand context the way a human fact-checker would — it is a statistical text pattern classifier, not a truth engine.
- Performance on genuinely novel, out-of-distribution news (very recent events, unfamiliar publication styles, short or ambiguous text) is likely lower than the 98.7% reported above.
- A production system would need continuous retraining on fresh data, source credibility signals, and ideally a human-in-the-loop review step for high-stakes classifications.

## Future Improvements

- Incorporate source/publisher metadata as an additional feature
- Experiment with more advanced models (e.g. fine-tuned transformer-based classifiers) and compare against this baseline
- Add explainability (e.g. highlighting which words most influenced a given prediction)
- Expand training data to include more recent and diverse news sources
