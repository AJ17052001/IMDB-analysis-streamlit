
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

st.set_page_config(page_title="Sentiment Analysis Dashboard")

@st.cache_data
def load_and_train():
    # Load the specific file
    # quoting=csv.QUOTE_ALL or QUOTE_MINIMAL helps handle reviews with commas
    df = pd.read_csv(
        "edited_csv_file.csv", 
        engine='python', 
        on_bad_lines='skip',
        quoting=csv.QUOTE_MINIMAL
    )
    
    # Preprocessing
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df.dropna(subset=['review', 'sentiment'])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    # Metrics
    y_pred = nb.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, nb, acc, cm

# --- UI Setup ---
st.title("Movie Review Sentiment Classifier")

try:
    tfidf, nb, acc, cm = load_and_train()

    st.write(f"**Model Accuracy:** {acc:.4f}")

    # Layout for Matrix and Prediction
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        

    with col2:
        st.subheader("Test a Review")
        user_review = st.text_input("Enter review text:")
        if user_review:
            vec = tfidf.transform([user_review])
            prediction = nb.predict(vec)[0]
            prob = nb.predict_proba(vec)[0]
            
            label = "Positive" if prediction == 1 else "Negative"
            confidence = prob[1] if prediction == 1 else prob[0]
            
            st.info(f"Sentiment: **{label}**")
            st.write(f"Confidence: {confidence:.2%}")

except FileNotFoundError:
    st.error("File 'edited_csv_file.csv' not found. Please ensure the file is in the application folder.")
except Exception as e:
    st.error(f"An error occurred: {e}")
