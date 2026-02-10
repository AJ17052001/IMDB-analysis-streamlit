
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Page config
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("IMDB Sentiment Analysis Tool")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your imdb_review_dataset (CSV or CSV.GZ)", type=['csv', 'gz'])

# We use cache_data to prevent re-running the training logic on every interaction
@st.cache_data
def process_and_train(file):
    # Load data
    df = pd.read_csv(file, compression='infer')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
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
    
    return tfidf, nb, acc, cm, df.head()

if uploaded_file:
    # Run pipeline
    tfidf, nb, acc, cm, preview_df = process_and_train(uploaded_file)
    
    # --- UI Layout ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(preview_df)
        st.write(f"**Model Accuracy:** {acc:.4f}")
        
        st.subheader("Test a Review")
        user_review = st.text_input("Enter review text:", placeholder="The movie was great!")
        if user_review:
            vec = tfidf.transform([user_review])
            prediction = nb.predict(vec)[0]
            label = "Positive" if prediction == 1 else "Negative"
            color = "green" if prediction == 1 else "red"
            st.markdown(f"Predicted Sentiment: :{color}[**{label}**]")

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("Please upload the IMDB Dataset to begin.")
