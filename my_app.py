import streamlit as st
from io import StringIO
import torch
from transformers import BertForSequenceClassification, BertTokenizer

def load_model():
    model = BertForSequenceClassification.from_pretrained("olesya2096/my_bert_imdb")
    return model

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('olesya2096/my_bert_imdb')
    return tokenizer

def print_ans(predictions):
    rate = 'positive'
    if predictions == 0:
        res = '1'
        rate = 'negative'
    elif predictions == 1:
        res = '2'
        rate = 'negative'
    elif predictions == 2:
        res = '3'
        rate = 'negative'
    elif predictions == 3:
        res = '4'
        rate = 'negative'
    elif predictions == 4:
        res = '7'
    elif predictions == 5:
        res = '8'
    elif predictions == 6:
        res = '9'
    elif predictions == 7:
        res = '10'
    ans = rate + ', rate is ' + res
    return ans

def load_file():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        st.write(string_data)
        return string_data
    else:
        return None


st.title('Классификация отзывов')
text = load_file()
result = st.button('Классифицировать отзыв')
model = load_model()
tokenizer = load_tokenizer()

if result:
    encoded_text = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_text)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).flatten()
    st.write('**Результаты распознавания:**')
    res = print_ans(predictions)
    print(st.write(res))
