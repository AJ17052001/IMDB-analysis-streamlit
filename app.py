
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard")

@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("compressed_data.csv.gz", compression='gzip')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Generate Confusion Matrix data
    y_pred = model.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    
    return tfidf, model, cm

# App Interface
st.title("Sentiment Analysis & Model Evaluation")

tfidf, model, cm = load_and_train()

# Sidebar Navigation
page = st.sidebar.selectbox("Choose a Page", ["Predictor", "Model Metrics"])

if page == "Predictor":
    st.header("Live Prediction")
    user_input = st.text_area("Enter a movie review:")
    
    if st.button("Analyze"):
        if user_input:
            vec_input = tfidf.transform([user_input])
            pred = model.predict(vec_input)[0]
            prob = model.predict_proba(vec_input)[0]
            
            if pred == 1:
                st.success(f"Positive ({prob[1]:.2%})")
            else:
                st.error(f"Negative ({prob[0]:.2%})")
        else:
            st.warning("Please enter text.")

elif page == "Model Metrics":
    st.header("Confusion Matrix")
    st.write("This matrix shows how the model performed on the 20% test split.")
    
    # Plotting the Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Display plot in Streamlit
    st.pyplot(fig)
