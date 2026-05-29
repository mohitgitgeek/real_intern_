"""Internship Email Fraud Detector — Streamlit app.

Frontend + backend (Streamlit), an ML model (TF-IDF + RandomForest) and now a
SQLite-backed history of every analysis (see db.py). Tokenization no longer
depends on NLTK's `punkt`, so it runs out of the box.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re
import requests
from io import StringIO

import db

# Stopwords: use NLTK's list if present, otherwise a small built-in fallback so
# the app has no hard NLTK runtime dependency.
try:
    from nltk.corpus import stopwords as _nltk_stop
    STOP_WORDS = set(_nltk_stop.words('english'))
except Exception:
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'is', 'are', 'was',
        'were', 'be', 'been', 'to', 'of', 'in', 'on', 'for', 'with', 'as', 'at',
        'by', 'this', 'that', 'it', 'you', 'your', 'we', 'our', 'i', 'me', 'my',
    }


@st.cache_data
def load_data():
    try:
        url = 'https://raw.githubusercontent.com/manchanda2612/Email-Spam-Detection/master/emails.csv'
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text), names=['text', 'label'])
    except Exception as e:
        st.warning(f"Could not load data from URL: {e}")

    # Fallback: a small but balanced sample dataset.
    sample_data = {
        'text': [
            "Congratulations! You've been selected for our paid internship program.",
            "Dear candidate, please submit your resume for the software engineering position.",
            "URGENT: Send $100 registration fee for guaranteed internship position!!!",
            "Welcome to our company's 2024 internship program. Please complete the application.",
            "Send bank details now to receive internship stipend in advance!!",
            "We reviewed your application and would like to schedule an interview next week.",
            "Pay a small processing fee via wire transfer to confirm your internship seat.",
            "Thank you for applying. Our HR team will contact you with the next steps.",
            "Winning candidate! Claim your stipend now by sharing your card OTP immediately.",
            "Please find attached the offer letter for the summer internship role.",
        ],
        'label': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
    }
    df = pd.DataFrame(sample_data)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = re.findall(r'[a-z]+', text)          # simple, dependency-free tokenizer
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)


@st.cache_resource
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)
    return vectorizer, model


def predict_fraud(text, vectorizer, model):
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)
    probability = model.predict_proba(text_vectorized)[0]
    return prediction[0], probability


INDICATORS = {
    "urgent": "Contains urgent language", "bank": "Mentions bank details",
    "payment": "Requests payment", "fee": "Requests fees",
    "guarantee": "Makes guarantees", "immediate": "Demands immediate action",
    "@gmail.com": "Uses personal email domain", "wire": "Mentions wire transfers",
    "crypto": "Mentions cryptocurrency", "winning": "Uses lottery-like language",
    "otp": "Asks for OTP/one-time password",
}


def detect_risk_factors(email_text):
    low = email_text.lower()
    return [msg for kw, msg in INDICATORS.items() if kw in low]


def analyze_tab(vectorizer, model, X_test, y_test):
    st.write("""
    This application helps detect potentially fraudulent internship email announcements.
    Enter the email text below to analyze it. Every analysis is saved to a local history.
    """)
    email_text = st.text_area("Enter the internship email text:", height=200)

    if st.button("Analyze Email"):
        if not email_text.strip():
            st.warning("Please enter some email text to analyze.")
            return
        with st.spinner("Analyzing email..."):
            prediction, probability = predict_fraud(email_text, vectorizer, model)
            label = "Fraudulent" if prediction == 1 else "Legitimate"
            confidence = (probability[1] if prediction == 1 else probability[0]) * 100
            risk_factors = detect_risk_factors(email_text)

            db.save_analysis(email_text, label, confidence, risk_factors)

            st.header("Analysis Results")
            if prediction == 1:
                st.error("⚠️ This email appears to be potentially fraudulent!")
            else:
                st.success("✅ This email appears to be legitimate.")
            st.write(f"Confidence: {confidence:.2f}%")

            st.subheader("Risk Factors Detected:")
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("No obvious risk factors detected.")

            st.subheader("Recommendations:")
            st.write("""
            - Verify the sender's email domain matches the company
            - Check the company's official website for the internship listing
            - Never send payment or sensitive personal information via email
            - Contact the company's HR department through official channels
            - Be wary of opportunities that seem too good to be true
            """)

    if st.checkbox("Show Model Performance Metrics"):
        st.subheader("Model Performance Metrics")
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)
        st.code(classification_report(y_test, y_pred, zero_division=0))


def history_tab():
    st.header("Analysis History")
    s = db.stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total analyses", s['total'])
    col2.metric("Flagged fraudulent", s['fraud'])
    col3.metric("Legitimate", s['legit'])

    rows = db.get_history()
    if rows:
        df = pd.DataFrame(rows)
        df['confidence'] = df['confidence'].round(1)
        df = df.rename(columns={
            'created_at': 'Time (UTC)', 'snippet': 'Email (snippet)',
            'prediction': 'Verdict', 'confidence': 'Confidence %', 'risk_factors': 'Risk factors',
        })
        st.dataframe(df, use_container_width=True)
        if st.button("Clear history"):
            db.clear_history()
            st.rerun()
    else:
        st.info("No analyses yet — run one from the Analyze tab.")


def main():
    st.title("🕵️ Internship Email Fraud Detector")
    db.init_db()

    with st.spinner("Loading and preparing the model..."):
        df = load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].apply(preprocess_text), df['label'],
            test_size=0.2, random_state=42,
        )
        vectorizer, model = train_model(X_train, y_train)

    tab1, tab2 = st.tabs(["🔍 Analyze", "📜 History"])
    with tab1:
        analyze_tab(vectorizer, model, X_test, y_test)
    with tab2:
        history_tab()


if __name__ == "__main__":
    main()
