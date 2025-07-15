# app.py
import streamlit as st
import joblib
import csv
import os


# Load model and vectorizer
model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')


# Streamlit UI

st.title("📩 Spam Message Classifier")
st.subheader("Detect whether a message is SPAM or NOT SPAM")

user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        vect_msg = vectorizer.transform([user_input])
        prediction = model.predict(vect_msg)[0]
        result = "🚫 SPAM" if prediction else "✅ Not Spam"
        st.success(f"Prediction: {result}")


