import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re

# Load your dataset
@st.cache_data
def load_data():
    data = pd.read_csv('anime_info_all_pages.csv') 
    return data

data = load_data()

# Preprocessing function for Arabic text
def preprocess_arabic_text(text):
    if not isinstance(text, str):
        return ""  # Return empty string for non-string entries
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Vectorizing the stories using TF-IDF
@st.cache_data
def vectorize_stories(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['story'].apply(preprocess_arabic_text))
    return tfidf_vectorizer, tfidf_matrix

tfidf_vectorizer, tfidf_matrix = vectorize_stories(data)

# Function to recommend series or movies
def recommend_series_or_movie(story, top_n=5):
    processed_story = preprocess_arabic_text(story)
    input_vec = tfidf_vectorizer.transform([processed_story])
    cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    similar_indices = np.argsort(-cosine_similarities)[:top_n]
    recommendations = data.iloc[similar_indices][['title', 'story', 'picture']]
    return recommendations

# Streamlit interface
st.title('Anime Recommendation System')
st.write('Enter a story description and receive anime recommendations!')

# Text input for the story
user_input_story = st.text_area("Story Description", "Type or paste the story here...")

# Streamlit interface for displaying recommendations
if st.button('Recommend'):
    with st.spinner('Finding similar animes...'):
        recommendations = recommend_series_or_movie(user_input_story)
        
        # Display each recommendation with title, story, and picture
        for idx, row in recommendations.iterrows():
            st.subheader(row['title'])
            st.write(row['story'])
            st.image(row['picture'])
# Credits
st.sidebar.header("Project Team")
st.sidebar.write("Nawaf Binsaad")
st.sidebar.write("Mohammed Adel")
st.sidebar.write("Abdulaziz Khoja")