import streamlit as st
#from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Load the sentiment analysis pipeline
@st.cache(allow_output_mutation=True)
def load_model():
    return pipeline("sentiment-analysis")

def load_model_malay():
    #model_name = "huggingface/bert-base-bahasa-cased"
    model_name = "mesolitica/bert-base-standard-bahasa-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


model = load_model()

# Streamlit app
st.title("Sentiment Analysis with BERT")
st.write("Enter some text and the model will analyze the sentiment.")

# User input
user_input = st.text_area("Enter your text here:")

# Model inference
if st.button("Analyze"):
    if user_input:
        results = model(user_input)
        for result in results:
            st.write(f"**Label:** {result['label']}")
            st.write(f"**Score:** {result['score']:.4f}")
    else:
        st.write("Please enter some text to analyze.")
