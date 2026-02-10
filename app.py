



                    import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Title
# -------------------------------
st.title(" IMDB Movie Review Sentiment Analysis")
st.write("Classifies movie reviews as **Positive** or **Negative** using Naive Bayes.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Train-Test Split
# -------------------------------
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -------------------------------
# Train Model
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Results
# -------------------------------
st.subheader(" Model Performance")

st.write(f"**Accuracy:** {accuracy:.4f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='mako',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive'],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Sentiment Analysis Confusion Matrix")

st.pyplot(fig)

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("📝 Try Your Own Review")

user_review = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_tfidf = tfidf.transform([user_review])
        prediction = nb_model.predict(review_tfidf)[0]

        if prediction == 1:
            st.success(" Positive Review")
        else:
            st.error(" Negative Review")
