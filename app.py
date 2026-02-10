
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

st.set_page_config(page_title="Sentiment Analysis")

@st.cache_resource
def load_and_train():
    # Using 'sep' and 'quoting' to fix parsing issues that cause single-class errors
    df = pd.read_csv(
        "edited_csv_file.csv", 
        on_bad_lines='skip',
        quoting=csv.QUOTE_MINIMAL,
        engine='python'
    )
    
    # Drop rows with missing values
    df = df.dropna(subset=['review', 'sentiment'])
    
    # Standardize sentiment labels
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Final drop for any mapping failures
    df = df.dropna(subset=['sentiment'])
    
    # CRITICAL CHECK: Ensure we have both 0 and 1
    if len(df['sentiment'].unique()) < 2:
        st.error("The dataset only contains one class. Check your CSV formatting!")
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    y_pred = nb.predict(tfidf.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, nb, acc, cm

st.title("Sentiment Classifier")

try:
    tfidf, nb, acc, cm = load_and_train()

    if nb:
        st.write(f"Model Accuracy: {acc:.4f}")
        
        user_review = st.text_input("Test a Review")
        if user_review:
            vec = tfidf.transform([user_review])
            prediction = nb.predict(vec)[0]
            
            # Safely handle probability display
            probs = nb.predict_proba(vec)[0]
            
            if prediction == 1:
                st.success(f"Positive Sentiment")
                if len(probs) > 1: st.write(f"Confidence: {probs[1]:.2%}")
            else:
                st.error(f"Negative Sentiment")
                if len(probs) > 1: st.write(f"Confidence: {probs[0]:.2%}")

        st.divider()
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax)
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
