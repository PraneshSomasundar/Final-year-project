import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
url = "./Dataset/cybertroll_dataset.csv"
text = "content"
flag = "annotation"
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
# Train the Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train_vectorized, y_train)
accuracy = clf.score(X_test_vectorized, y_test)

# Save the trained classifier to a pickle file
with open("./pickle/rf_classifier.pkl", "wb") as f:
    pickle.dump((clf,accuracy), f)

# Save the TF-IDF vectorizer to a pickle file
with open("./pickle/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
