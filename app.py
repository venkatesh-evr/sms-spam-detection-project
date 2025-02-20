import nltk 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    vectorizer = pickle.load(open("C:/Users/ASUS/sms-spam-detection-project/vectorized.pkl", 'rb'))
    model = pickle.load(open("C:/Users/ASUS/sms-spam-detection-project/model.pkl", 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

st.title("SMS Spam Detection Model")
st.write("Developed by venkatesh-evr")

input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
