import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from io import StringIO

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the Fraudulent Email Dataset
@st.cache_data
def load_data():
    try:
        # First attempt: Try loading from the URL
        url = 'https://raw.githubusercontent.com/manchanda2612/Email-Spam-Detection/master/emails.csv'
        response = requests.get(url)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text), names=['text', 'label'])
    except Exception as e:
        st.warning(f"Could not load data from URL: {str(e)}")
    
    try:
        # Second attempt: Try loading from local file
        return pd.read_csv('emails.csv', names=['text', 'label'])
    except Exception as e:
        st.error("Could not load data from local file. Using sample dataset instead.")
        
        # Fallback: Create a small sample dataset
        sample_data = {
            'text': [
                "Congratulations! You've been selected for our paid internship program.",
                "Dear candidate, please submit your resume for the software engineering position.",
                "URGENT: Send $100 registration fee for guaranteed internship position!!!",
                "Welcome to our company's 2024 internship program. Please complete the application.",
                "Send bank details now to receive internship stipend in advance!!",
            ],
            'label': ['ham', 'ham', 'spam', 'ham', 'spam']
        }
        df = pd.DataFrame(sample_data)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        return df

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Join tokens back into text
    return ' '.join(tokens)

@st.cache_resource
def train_model(X_train, y_train):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    return vectorizer, model

def predict_fraud(text, vectorizer, model):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    # Vectorize the text
    text_vectorized = vectorizer.transform([processed_text])
    # Make prediction
    prediction = model.predict(text_vectorized)
    probability = model.predict_proba(text_vectorized)[0]
    return prediction[0], probability

# Main Streamlit app
def main():
    st.title("Internship Email Fraud Detector")
    st.write("""
    This application helps detect potentially fraudulent internship email announcements.
    Enter the email text below to analyze it.
    """)
    
    # Load and prepare data
    with st.spinner("Loading and preparing the model..."):
        df = load_data()
        
        # Train model (in real application, this should be done offline and loaded from disk)
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].apply(preprocess_text),
            df['label'],
            test_size=0.2,
            random_state=42
        )
        vectorizer, model = train_model(X_train, y_train)
    
    # Create text input area
    email_text = st.text_area("Enter the internship email text:", height=200)
    
    if st.button("Analyze Email"):
        if email_text:
            with st.spinner("Analyzing email..."):
                # Make prediction
                prediction, probability = predict_fraud(email_text, vectorizer, model)
                
                # Display results
                st.header("Analysis Results")
                
                if prediction == 1:
                    st.error("⚠️ This email appears to be potentially fraudulent!")
                    confidence = probability[1] * 100
                else:
                    st.success("✅ This email appears to be legitimate.")
                    confidence = probability[0] * 100
                    
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Display risk factors
                st.subheader("Risk Factors Detected:")
                risk_factors = []
                
                # Enhanced fraud indicators
                indicators = {
                    "urgent": "Contains urgent language",
                    "bank": "Mentions bank details",
                    "payment": "Requests payment",
                    "fee": "Requests fees",
                    "guarantee": "Makes guarantees",
                    "immediate": "Demands immediate action",
                    "@gmail.com": "Uses personal email domain",
                    "wire": "Mentions wire transfers",
                    "crypto": "Mentions cryptocurrency",
                    "winning": "Uses lottery-like language"
                }
                
                for keyword, message in indicators.items():
                    if keyword in email_text.lower():
                        risk_factors.append(message)
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.write("No obvious risk factors detected.")
                
                # Show recommendations
                st.subheader("Recommendations:")
                st.write("""
                - Verify the sender's email domain matches the company
                - Check the company's official website for the internship listing
                - Never send payment or sensitive personal information via email
                - Contact the company's HR department through official channels
                - Research the company on professional networks like LinkedIn
                - Be wary of opportunities that seem too good to be true
                """)
        else:
            st.warning("Please enter some email text to analyze.")
    
    # Add model performance metrics
    if st.checkbox("Show Model Performance Metrics"):
        st.subheader("Model Performance Metrics")
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)
        st.code(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()