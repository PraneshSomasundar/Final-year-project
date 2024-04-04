import streamlit as st
from PIL import Image
import pickle
import string
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import dotenv_values
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

env_vars = dotenv_values(".env")

class YouTubeAPI:
    def __init__(self):
        self.api_service_name = "youtube"
        self.api_version = "v3"
        self.DEVELOPER_KEY = env_vars["ACCESS_TOKEN"]
        self.youtube = googleapiclient.discovery.build(
            self.api_service_name, self.api_version, developerKey=self.DEVELOPER_KEY)

    def get_comments(self, video_id, max_results=100):
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results
        )
        response = request.execute()
        data = []
        for item in response.get('items', []):
            data.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
        return data
    
    
    def get_video_title(self, video_id):
        request = self.youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        video_title = response['items'][0]['snippet']['title']
        return video_title
    


hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
</style>
"""

showWarningOnDirectExecution = False
ps = PorterStemmer()
image = Image.open('.\\icons\\logo.png')

st.set_page_config(page_title="Cyberbullying Detection", page_icon=image)
st.markdown(hide_menu, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(image, use_column_width=True, output_format='auto')
st.sidebar.markdown("---")
st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2024 | Secure Net</h1>", unsafe_allow_html=True)

youtube_api = YouTubeAPI()

st.title("Cyber-Bullying Detection🔍")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

video_id = st.text_input("Enter the YouTube Video ID:")
if video_id:
    try:
        comments = youtube_api.get_comments(video_id)
        title = youtube_api.get_video_title(video_id)

        def clean_text(tweet):
            tweet = re.sub(r'http\S+', '', tweet)
            tweet = re.sub(r'@[^\s]+[\s]?','',tweet)
            tweet = re.sub(r'#[^\s]+[\s]?','',tweet)
            tweet = re.sub(r':[^\s]+[\s]?','',tweet)
            tweet = re.sub('[^ a-zA-Z0-9]' , '', tweet)
            tweet = re.sub('RT' , '', tweet)
            tweet = re.sub('[0-9]', '', tweet)
            return tweet

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

        # Load the pickled classifier and vectorizer
        with open("./pickle/rf_classifier.pkl", "rb") as f:
            clf, accuracy = pickle.load(f)

        with open("./pickle/tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        st.markdown("---")
        st.subheader(title)
        
        st.markdown("---")
        st.subheader("Comments")
        # Display all comments
        for comment in comments:
            cleanText = clean_text(comment)
            transformText = transform_text(cleanText)
            vector_input = vectorizer.transform([transformText])
            result = clf.predict(vector_input)[0]
            if result == 1:
                st.error(comment)
            else:
                st.success(comment)

        st.markdown("---")

        st.sidebar.markdown("---")
        st.sidebar.markdown("---")
        st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2024 | Secure Net</h1>", unsafe_allow_html=True)

        st.subheader("Model Accuracy")
        expander_accuracy = st.expander("Information", expanded=False)
        with expander_accuracy:
            st.info("Model Accuracy using Random Forest (RF) Classifier!")
        st.warning(f"Accuracy:  **{round(accuracy * 100, 2)} %**")
        st.markdown("---")
    except Exception as e:
        st.error("Error fetching comments. Please enter a valid YouTube video ID.")
