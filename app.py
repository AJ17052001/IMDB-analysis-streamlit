



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="wide")

@st.cache_resource
def train_model(file_path):
    # Load with settings to handle reviews containing commas/quotes
    df = pd.read_csv(
        file_path, 
        on_bad_lines='skip', 
        quoting=csv.QUOTE_MINIMAL, 
        engine='python'
    )
    
    # Clean column names (removes hidden spaces)
    df.columns = df.columns.str.strip()
    
    # Aggressive cleaning of the sentiment column
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Drop rows that failed to map or have empty reviews
    df = df.dropna(subset=['review', 'sentiment'])

    # Validate classes
    unique_classes = df['sentiment'].unique()
    if len(unique_classes) < 2:
        return None, None, None, None, df # Return df for debugging

    # Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    # Calculate Metrics
    y_pred = nb.predict(tfidf.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, nb, acc, cm, df

# --- UI Layout ---
st.title("IMDB Movie Review Classifier")

try:
    tfidf, model, acc, cm, processed_df = train_model("edited_csv_file.csv")

    if model is not None:
        st.success(f"Model trained successfully! Accuracy: {acc:.2%}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Test the Model")
            user_input = st.text_area("Enter a movie review:", height=150)
            if st.button("Predict Sentiment"):
                if user_input:
                    vec = tfidf.transform([user_input])
                    prediction = model.predict(vec)[0]
                    probs = model.predict_proba(vec)[0]
                    
                    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
                    confidence = probs[1] if prediction == 1 else probs[0]
                    
                    st.metric("Prediction", label, delta=f"{confidence:.2%} Confidence")
                else:
                    st.warning("Please enter some text.")

        with col2:
            st.subheader("Performance: Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            

   
