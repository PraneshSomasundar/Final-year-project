import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def logistic_regression(data,text,flag):    
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenization
        tokens = nltk.word_tokenize(text)
        # Remove stopwords and perform stemming
        ps = PorterStemmer()
        tokens = [ps.stem(word) for word in tokens if word not in set(stopwords.words('english'))]
        return ' '.join(tokens)

    # Apply preprocessing to the 'comments' column
    data['clean_comments'] = data[text].apply(preprocess_text)

    # Feature extraction
    tfidf = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
    X = tfidf.fit_transform(data['clean_comments'])
    y = data[flag]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select a machine learning algorithm (Logistic Regression in this example)
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

dataset_info = {
    "Cyber Bullying Types Dataset": {
        "url": "./Dataset/CyberBullyingTypesDataset.csv",
        "text": "Tweet",
        "flag": "Class" # vectorize
    },
    "Cyber Troll Dataset": {
        "url": "./Dataset/cybertroll_dataset.csv",
        "text": "content",
        "flag": "annotation"
    },
    "Classified Tweets Dataset": {
        "url": "./Dataset/classified_tweets.csv",
        "text": "text",
        "flag": "cyberbullying"
    },
    "Cyberbullying Dataset": {
        "url": "./Dataset/cyberbullying.csv",
        "text": "tweet_text",
        "flag": "cyberbullying_type"
    }
}

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
image = Image.open('icons/logo.png')


st.set_page_config(page_title = "Algorithms", page_icon = image)

st.markdown(hide_menu, unsafe_allow_html=True)

 
st.sidebar.image(image , use_column_width=True, output_format='auto')


st.sidebar.markdown("---")


st.sidebar.markdown(" <br> <br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | Ioannis Bakomichalis</h1>", unsafe_allow_html=True)




st.title("Algorithms📊")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

all_Datasets = ["Select a Dataset", "Cyber Troll Dataset","Classified Tweets Dataset","Cyberbullying Dataset"] # ,"Cyber Bullying Types Dataset"]
data_choice = st.selectbox("Dataset", all_Datasets)
all_Vectorizers = ["Select a Vectorizer", "TF-IDF"]
vect_choice = st.selectbox("Vectorizer", all_Vectorizers)
all_ML_models = ["Select a Machine Learning Algorithm", "Compare All", "Logistic Regression", "Naive Bayes", "Random Forest", "K Nearest Neighbors", "SGD Classifier"]
model_choice = st.selectbox("Machine Learning Algorithm", all_ML_models)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

if data_choice == "Select a Dataset" and vect_choice != "Select a Vectorizer" and model_choice != "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Dataset_**]")
elif data_choice != "Select a Dataset" and vect_choice == "Select a Vectorizer" and model_choice != "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Vectorizer_**]")
elif data_choice != "Select a Dataset" and vect_choice != "Select a Vectorizer" and model_choice == "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Machine Learning Algorithm_**]")
elif data_choice == "Select a Dataset" and vect_choice == "Select a Vectorizer" and model_choice != "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Dataset_** and **_Vectorizer_**]")
elif data_choice == "Select a Dataset" and vect_choice != "Select a Vectorizer" and model_choice == "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Dataset_** and **_Machine Learning Algorithm_**]")
elif data_choice != "Select a Dataset" and vect_choice == "Select a Vectorizer" and model_choice == "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Vectorizer_** and **_Machine Learning Algorithm_**]")
elif data_choice == "Select a Dataset" and vect_choice == "Select a Vectorizer" and model_choice == "Select a Machine Learning Algorithm":
    st.warning(":white[You should select **_Dataset_** and **_Vectorizer_** and **_Machine Learning Algorithm_**]")
else:
    # Load the dataset
    url = dataset_info[data_choice]["url"]
    text = dataset_info[data_choice]["text"]
    flag = dataset_info[data_choice]["flag"]
    data = pd.read_csv(url)

    # Preprocessing
    data.dropna(inplace=True)  # Drop any rows with missing values
    X = data[text]
    y = data[flag]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text using TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize classifiers
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "K Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SGD Classifier": SGDClassifier()
    }

    if model_choice == "Compare All":
        # Train and evaluate classifiers
        results = {}
        for name, clf in classifiers.items():
            clf.fit(X_train_vectorized, y_train)
            y_pred = clf.predict(X_test_vectorized)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        results["Logistic Regression"] = logistic_regression(data, text, flag)
        # Print results
        for name, acc in results.items():
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader(f"Model: {name}")
            st.success(f":green[Accuracy: **{round(acc * 100, 2)} %**]")

    elif model_choice == "Logistic Regression":
        accuracy = logistic_regression(data, text, flag)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Model: {model_choice}")
        st.success(f":green[Accuracy: **{round(accuracy * 100, 2)} %**]")
    else:
        clf = classifiers[model_choice]
        clf.fit(X_train_vectorized, y_train)
        y_pred = clf.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Model: {model_choice}")
        st.success(f":green[Accuracy: **{round(accuracy * 100, 2)} %**]")
