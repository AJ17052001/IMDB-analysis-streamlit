


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import csv

# Set page layout
st.set_page_config(page_title="Sentiment Analysis Pro", layout="wide")

@st.cache_resource
def build_pipeline(file_path):
    # 1. Load data with high tolerance for text artifacts
    try:
        df = pd.read_csv(
            file_path, 
            on_bad_lines='skip', 
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\',
            engine='python'
        )
    except Exception as e:
        return None, f"Read Error: {e}"

    # 2. Clean column names and drop empty rows
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['review', 'sentiment'])

    # 3. Robust Label Mapping
    # Standardize to lowercase and remove spaces so " positive" matches "positive"
    df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
    mapping = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(mapping)
    
    # Drop rows that didn't map correctly
    df = df.dropna(subset=['label'])

    # 4. Check for Class Imbalance
    if len(df['label'].unique()) < 2:
        return None, "Dataset Error: Only one class found (check CSV formatting)."

    # 5. Model Training
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # 6. Evaluation
    y_pred = model.predict(tfidf.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return (tfidf, model, acc, cm), None

# --- Main UI ---
st.title("Sentiment Analysis Dashboard")
st.markdown("---")

# Attempt to load and train
data_package, error_msg = build_pipeline("edited_csv_file.csv")

if error_msg:
    st.error(error_msg)
    st.info("Check if your CSV has 'review' and 'sentiment' columns with 'positive'/'negative' values.")
else:
    tfidf, model, acc, cm = data_package
    
    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{acc:.2%}")
        
        # Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        

    with col2:
        st.subheader("Live Prediction")
        user_text = st.text_area("Paste a movie review here:", height=200)
        
        if st.button("Run Sentiment Analysis"):
            if user_text:
                # Transform and Predict
                vec = tfidf.transform([user_text])
                prediction = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]

                # Result Display
                if prediction == 1:
                    st.success(f"Result: POSITIVE")
                    st.progress(probs[1])
                    st.write(f"Confidence: {probs[1]:.2%}")
                else:
                    st.error(f"Result: NEGATIVE")
                    st.progress(probs[0])
                    st.write(f"Confidence: {probs[0]:.2%}")
            else:
                st.warning("Please enter text to analyze.")
