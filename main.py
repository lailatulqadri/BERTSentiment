import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizerFast

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return model, tokenizer

def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
    return 'Positive' if probs[1] > probs[0] else 'Negative'

model, tokenizer = load_model()

st.title('BERT Sentiment Analysis')

text = st.text_input('Enter text here')

if text:
    sentiment = predict_sentiment(model, tokenizer, text)
    st.write(f'Sentiment: {sentiment}')
