import streamlit as st
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# NER model and tokenizer
model_name = "debabrata-ai/NepNER"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("debabrata-ai/NepNER")

ner_labels = model.config.id2label

# title
st.title("Nepali NER Prediction App ")

# User input
user_input = st.text_area("Enter a sentence or paragraph:")

if st.button("Predict NER"):
    if user_input:
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(user_input)))
        inputs = tokenizer.encode(user_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model(inputs).logits

        predictions = torch.argmax(outputs, dim=2).squeeze().tolist()
        ner_tags = [ner_labels[label_id] for label_id in predictions]

        st.subheader("NER Predictions:")
        for token, tag in zip(tokens, ner_tags):
            st.write(f"{token}: {tag}")
    else:
        st.warning("Please enter a sentence or paragraph for NER prediction.")

