
# app.py - Streamlit UI for Fake News Detection using BERT

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load("bert_fake_news_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# UI Layout
st.title("📰 Fake News Detector - BERT")
st.markdown("Enter any news article text below to check if it's **Real** or **Fake**.")

user_input = st.text_area("Enter News Content:")

if st.button("Check News"):
    if user_input:
        prediction = predict(user_input)
        if prediction == 1:
            st.success("✅ This news appears to be **Real**.")
        else:
            st.error("🚫 This news appears to be **Fake**.")
    else:
        st.warning("Please enter some news content to check.")
