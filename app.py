
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

st.set_page_config(page_title="Sentiment Analysis Fix")

@st.cache_resource
def load_and_train():
    # 1. Load with flexible parsing
    df = pd.read_csv(
        "edited_csv_file.csv", 
        on_bad_lines='warn', 
        quoting=csv.QUOTE_MINIMAL,
        escapechar='\\'
    )
    
    # 2. Debug: Show user what the columns actually look like
    # st.write("Detected Columns:", df.columns.tolist())
    
    # 3. Aggressive sentiment cleaning
    # This handles cases like 'positive ', 'POSITIVE', or even ' positive'
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    
    # Check if the labels are actually there before mapping
    unique_labels = df['sentiment'].unique()
    
    # 4. Final Mapping
    df['sentiment_mapped'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Remove rows where mapping failed or text is missing
    df = df.dropna(subset=['review', 'sentiment_mapped'])
    
    if len(df['sentiment_mapped'].unique()) < 2:
        st.error(f"Found labels: {unique_labels}. Need 'positive' and 'negative'.")
        return None

    # 5. Training Pipeline
    X = df['review']
    y = df['sentiment_mapped'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    y_pred = nb.predict(tfidf.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, nb, acc, cm

st.title("Sentiment Analysis (Fixed Loader)")

try:
    tfidf, nb, acc, cm = load_and_train()

    if nb is not None:
        st.success(f"Model trained successfully! Accuracy: {acc:.4f}")
        
        # User Interaction
        review = st.text_input("Enter a review to test:")
        if review:
            vec = tfidf.transform([review])
            res = nb.predict(vec)[0]
            label = "Positive" if res == 1 else "Negative"
            st.info(f"Result: {label}")

        # Visualization
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        st.pyplot(fig)
        

except Exception as e:
    st.error(f"Critical Error: {e}")
