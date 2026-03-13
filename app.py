import streamlit as st
import joblib
import re
import altair as alt

# Load model and vectorizer exported from notebook
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub("\s+", " ", text)
    return text

def predict_language(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

# Title
st.title("Kenyan Language Identification System")
st.write("Detects Swahili, Sheng, English, and Luo in short text")

# User input
text_input = st.text_area("Enter text here:")

# Predict button
if st.button("Predict Language"):
    if text_input.strip() != "":
        result = predict_language(text_input)
        st.success(f"Predicted Language: {result.capitalize()}")
    else:
        st.warning("Please enter some text to predict")

