import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Tool")

# Use cache_resource to ensure the model only trains once per session
@st.cache_resource
def setup_model():
    try:
        # Loading compressed dataset
        df = pd.read_csv("compressed_data.csv.gz", compression='gzip')
        
        # Preprocessing
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Feature Extraction
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(df['review'])
        y = df['sentiment']
        
        # Model Training
        model = MultinomialNB()
        model.fit(X, y)
        
        return tfidf, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# --- UI Setup ---
st.title("Sentiment Analysis Classifier")
st.write("This application uses a Multinomial Naive Bayes model to predict the sentiment of movie reviews.")

tfidf, model = setup_model()

if model is not None:
    # Text input area
    review_text = st.text_area("Input Review", height=150, placeholder="Enter the movie review text here...")

    # Predict button
    if st.button("Predict"):
        if review_text.strip():
            # Process input
            processed_input = tfidf.transform([review_text])
            prediction = model.predict(processed_input)[0]
            probability = model.predict_proba(processed_input)[0]

            # Output results
            st.divider()
            if prediction == 1:
                st.success(f"Result: Positive Sentiment")
                st.info(f"Confidence: {probability[1]:.2%}")
            else:
                st.error(f"Result: Negative Sentiment")
                st.info(f"Confidence: {probability[0]:.2%}")
        else:
            st.warning("Please enter text before predicting.")

else:
    st.info("Please ensure 'IMDB Dataset.csv.gz' is in the same directory as this script.")

# Optional: Add a sidebar with project details
st.sidebar.header("About")
st.sidebar.write("Algorithm: Naive Bayes")
st.sidebar.write("Features: TF-IDF (5000 max features)")
