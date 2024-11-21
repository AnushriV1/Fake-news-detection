import streamlit as st
import pickle
import os

# Define the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), '../models')

# Load the model and vectorizer
try:
    with open(os.path.join(models_dir, 'naive_bayes.pkl'), 'rb') as f:
        nb_model = pickle.load(f)

    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit app
st.title("Fake News Detection")

# Input from user
news_title = st.text_input("Enter the news article title here:")
news_article = st.text_area("Enter the news article text here:", height=300)

if st.button("Predict"):
    if news_article:
        # Vectorize input
        input_vector = vectorizer.transform([news_article])
        
        # Get predictions from Naive Bayes model
        nb_pred = nb_model.predict(input_vector)[0]
        
        # Determine result text and color
        if nb_pred == 1:
            result = "Real"
            color = "green"
        else:
            result = "Fake"
            color = "red"
        
        # Display prediction with color
        st.subheader("Prediction:")
        st.markdown(f"<span style='color:{color}; font-size:24px;'>{result}</span>", unsafe_allow_html=True)

        st.write(f"Title: {news_title}")
    else:
        st.warning("Please enter a news article to predict.")
