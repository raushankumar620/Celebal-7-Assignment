# streamlit run app.py ( Run this command in terminal )
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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

# Load the saved model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error("Model files not found! Please make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display
        if result == 1:
            st.header("Spam")
            st.error("⚠️ This message is classified as SPAM!")
        else:
            st.header("Not Spam")
            st.success("✅ This message is NOT spam.")
            
        # Also show probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector_input)[0]
            st.write(f"Probability of being Ham: {proba[0]:.2%}")
            st.write(f"Probability of being Spam: {proba[1]:.2%}")
    else:
        st.warning("Please enter a message to classify.")

# Add some example messages for testing
st.sidebar.header("Example Messages")
st.sidebar.write("**Spam Example:**")
st.sidebar.write("Congratulations! You've won a $1000 gift card. Click here to claim your prize now!")

st.sidebar.write("**Ham Example:**")
st.sidebar.write("Hey, are you free this evening? Want to grab dinner?")

# Run this command in terminal:
# streamlit run app.py
