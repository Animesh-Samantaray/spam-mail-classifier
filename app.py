import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# Load vectorizer & model
cv = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ================== CUSTOM CSS ==================
st.markdown("""
    <style>
        /* Full background */
        .stApp {
            background: linear-gradient(135deg, #000428, #004e92); /* deep blue gradient */
            color: #ffffff;
        }
        .title {
            text-align: center;
            font-size:150px
            font-weight: bold;
            color: #00f5d4;
            text-shadow: 
                0 0 30px #00f5d4,
                0 0 60px #00f5d4,
                0 0 120px #00bbf9,
                0 0 180px #00f5d4,
                0 0 250px #00bbf9;
            animation: flicker 2.5s infinite alternate;
        }
        @keyframes flicker {
            0% { text-shadow: 0 0 20px #00f5d4, 0 0 40px #00bbf9, 0 0 60px #00f5d4; }
            50% { text-shadow: 0 0 50px #00f5d4, 0 0 100px #00bbf9, 0 0 150px #00f5d4; }
            100% { text-shadow: 0 0 40px #00bbf9, 0 0 80px #00f5d4, 0 0 120px #00bbf9; }
        }

        /* Text area styling */
        textarea {
            background: rgba(0, 0, 50, 0.6) !important;
            color: #fff !important;
            border-radius: 14px !important;
            border: 2px solid #00f5d4 !important;
            font-size: 18px !important;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 14px;
            font-size: 22px;
            background: linear-gradient(90deg, #00f5d4, #00bbf9);
            color: black;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.1);
            box-shadow: 0px 0px 30px #00f5d4;
        }

        /* Result box */
        .result-box {
            padding: 28px;
            border-radius: 20px;
            text-align: center;
            font-size: 32px;
            margin-top: 30px;
            font-weight: bold;
            box-shadow: 0px 0px 40px rgba(0, 245, 212, 0.9);
        }
        .spam {
            background: rgba(255, 0, 70, 0.15);
            border: 3px solid #ff004e;
            color: #ff004e;
        }
        .not-spam {
            background: rgba(0, 245, 212, 0.15);
            border: 3px solid #00f5d4;
            color: #00f5d4;
        }
    </style>
""", unsafe_allow_html=True)

# ================== APP TITLE ==================
st.markdown('<h1 class="title">📧 AI-Powered Email Spam Classifier</h1>', unsafe_allow_html=True)

# ================== INPUT ==================
input_mail = st.text_area('✍️ Enter your email below:', height=150, placeholder="Paste your email content here...")

# Stopwords & Stemmer
s = stopwords.words('english')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    ans = [ps.stem(i) for i in y if i not in s and i not in string.punctuation]

    return " ".join(ans)

# ================== PREDICTION ==================
if st.button('🚀 Check Email'):
    if input_mail.strip() == "":
        st.warning("⚠️ Please enter some text before checking!")
    else:
        # Preprocess
        transformed_mail = transform_text(input_mail)

        # Vectorize
        vector_input = cv.transform([transformed_mail]).toarray()

        # Add num_characters feature
        num_chars = len(nltk.word_tokenize(transformed_mail))
        vector_input = np.hstack((vector_input, [[num_chars]]))

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.markdown('<div class="result-box spam">🚨 SPAM DETECTED 🚨</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✅ SAFE: Not Spam</div>', unsafe_allow_html=True)
