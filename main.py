import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment = 'positive' if predicted_class > 0 else 'negative'
    return probabilities


model, tokenizer = load_model()

st.title('BERT Sentiment Analysis')

text = st.text_input('Enter text here')

if text:
    sentiment = predict_sentiment(model, tokenizer, text)
    st.write(f'Sentiment: {sentiment}')
