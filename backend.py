from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import joblib
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

app = FastAPI(title="Internship Checker API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnnouncementRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    is_genuine: bool
    confidence: float
    flags: List[str]
    detailed_analysis: dict

# Initialize global variables for model and vectorizer
model = None
vectorizer = None

def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    # Lowercasing
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def initialize_model():
    """Initialize and train the model with sample data."""
    global model, vectorizer
    
    # Sample data
    data = {
        "text": [
            "We offer a paid internship at XYZ Corp. Apply now with no fees!",
            "Work from home! Urgent hiring, pay a fee to get started.",
            "XYZ Corp is looking for interns. Send your CV to hr@xyzcorp.com",
            "Exciting internship! Just a small entry fee to begin.",
            "Summer internship program at Tech Corp. Stipend provided.",
            "Quick money! Pay registration fee for guaranteed internship.",
            "Join our engineering team as an intern. Monthly stipend included.",
            "Immediate placement after payment of processing charges.",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for genuine, 0 for fake
    }
    
    df = pd.DataFrame(data)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_tfidf, y_train)

def analyze_red_flags(text: str) -> List[str]:
    """Analyze text for common red flags."""
    red_flags = {
        'payment required': 'Requests for payment',
        'fee': 'Mentions of fees',
        'pay to': 'Requests for payment',
        'urgent': 'Urgency tactics',
        'guaranteed': 'Unrealistic guarantees',
        'immediate': 'Urgency tactics',
        'processing charge': 'Hidden charges',
        'registration charge': 'Hidden charges'
    }
    
    found_flags = []
    lower_text = text.lower()
    
    for flag, description in red_flags.items():
        if flag in lower_text:
            found_flags.append(description)
    
    return list(set(found_flags))  # Remove duplicates

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts."""
    initialize_model()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_announcement(request: AnnouncementRequest):
    """Analyze an internship announcement."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty announcement text")
    
    try:
        # Preprocess text
        cleaned_text = preprocess_text(request.text)
        
        # Get model prediction
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_tfidf)[0]
        confidence = model.predict_proba(text_tfidf)[0]
        
        # Get prediction confidence
        prediction_confidence = confidence[1] if prediction == 1 else confidence[0]
        
        # Analyze for red flags
        flags = analyze_red_flags(request.text)
        
        # Prepare detailed analysis
        detailed_analysis = {
            "text_length": len(request.text),
            "contains_email": bool(re.search(r'\S+@\S+', request.text)),
            "contains_url": bool(re.search(r'http\S+|www\S+', request.text)),
            "red_flags_count": len(flags),
            "model_prediction": float(prediction),
            "model_confidence": float(prediction_confidence)
        }
        
        return AnalysisResponse(
            is_genuine=bool(prediction),
            confidence=float(prediction_confidence),
            flags=flags,
            detailed_analysis=detailed_analysis
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)