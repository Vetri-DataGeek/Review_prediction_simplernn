import streamlit as st
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb

#step1 load imdb word index
word_index = imdb.get_word_index()
reversed_word_index = {v:k for k,v in word_index.items()}

# load the prediction model
movie_reviewer = load_model('simple_rnn_imdb.h5')

#function to preprocess the reveiw & prediction
def preprocess_text(review):
    words = review.lower().split()
    encoded_text = [word_index.get(word,2) for word in words]
    paded_review = sequence.pad_sequences([encoded_text],maxlen=500)
    return paded_review

def predict_sentiment(review):
    prediction = movie_reviewer.predict(review)
    sentiment_Score = prediction[0][0]
    sentiment = 'Positive' if sentiment_Score > 0.5 else 'Negative'
    return sentiment,sentiment_Score

#stream app creation

st.title("Moview review sentiment Analysis")
movie_name = st.text_input("Enter the Movie Name")
movie_review = st.text_input("Enter the your review about the Movie")

if st.button("Predict Sentiment"):
    preprocessed_review = preprocess_text(movie_review)
    sentiment,sentiment_score = predict_sentiment(preprocessed_review)
    st.write(f"Movie Name:{movie_name}")
    st.write(f"Sentiment Verdict: {sentiment}")
    st.write(f"Sentiment Score: {sentiment_score}")
else:
    st.write("Enter the movie name & review")

