import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import joblib
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources (with error handling)
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except:
        return False

# Load GloVe embeddings
@st.cache_resource
def load_glove_embeddings():
    """Load GloVe embeddings from file"""
    try:
        embeddings_dict = {}
        with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = vector
        return embeddings_dict
    except FileNotFoundError:
        st.error("GloVe embeddings file not found! Please ensure 'glove.6B.300d.txt' is in the same directory.")
        return None

# Load models
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    try:
        classifier = joblib.load('glove_emotion_classifier.pkl')
        scaler = joblib.load('glove_feature_scaler.pkl')
        tfidf_vectorizer = joblib.load('glove_tfidf_vectorizer.pkl')
        return classifier, scaler, tfidf_vectorizer
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None

# Text preprocessing function
def preprocess_text(text):
    """Complete text preprocessing pipeline"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' +', ' ', text).strip()
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# GloVe embedding function
def get_glove_embedding(tokens, embeddings_dict, embedding_dim=300):
    """Convert tokens to sentence embedding using GloVe"""
    embeddings = []
    found_words = 0
    
    for token in tokens:
        if token in embeddings_dict:
            embeddings.append(embeddings_dict[token])
            found_words += 1
        else:
            embeddings.append(np.zeros(embedding_dim))
    
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(embedding_dim)
    
    return sentence_embedding, found_words, len(tokens)

# Prediction function
def predict_emotion(lyrics_text, glove_dict, tfidf_vectorizer, scaler, classifier):
    """Predict emotion using GloVe + TF-IDF"""
    tokens = preprocess_text(lyrics_text)
    
    # GloVe embedding
    glove_features, found_words, total_words = get_glove_embedding(tokens, glove_dict)
    
    # TF-IDF features
    processed_text = ' '.join(tokens)
    tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray()[0]
    
    # Combine and predict
    combined_features = np.hstack([glove_features, tfidf_features])
    scaled_features = scaler.transform([combined_features])
    
    prediction = classifier.predict(scaled_features)[0]
    probabilities = classifier.predict_proba(scaled_features)[0] if hasattr(classifier, 'predict_proba') else None
    
    coverage = found_words / total_words if total_words > 0 else 0
    
    return prediction, probabilities, coverage

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Emotion Classification Dashboard",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Lyrics Emotion Classification Dashboard")
    st.markdown("Classify emotions in song lyrics using GloVe embeddings and TF-IDF features")
    
    # Initialize resources
    if not download_nltk_resources():
        st.warning("Some NLTK resources couldn't be downloaded. The app may not work properly.")
    
    # Load models and embeddings
    with st.spinner("Loading models and embeddings..."):
        glove_embeddings = load_glove_embeddings()
        classifier, scaler, tfidf_vectorizer = load_models()
    
    if glove_embeddings is None or classifier is None:
        st.error("Failed to load required files. Please ensure all model files and GloVe embeddings are available.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📊 Model Information")
        model_name = type(classifier).__name__
        st.write(f"**Model Type:** {model_name}")
        st.write(f"**Features:** GloVe (300) + TF-IDF")
        st.write(f"**Classes:** {len(classifier.classes_)}")
        
        st.header("🏷️ Emotion Classes")
        for i, emotion in enumerate(classifier.classes_):
            st.write(f"{i+1}. {emotion}")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🔤 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Single Lyrics Classification")
        
        # Text input
        lyrics_input = st.text_area(
            "Enter song lyrics:",
            placeholder="Type or paste your song lyrics here...",
            height=150
        )
        
        if st.button("🎭 Predict Emotion", type="primary"):
            if lyrics_input.strip():
                with st.spinner("Analyzing lyrics..."):
                    prediction, probabilities, coverage = predict_emotion(
                        lyrics_input, glove_embeddings, tfidf_vectorizer, scaler, classifier
                    )
                
                # Results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Prediction Result")
                    st.success(f"**Predicted Emotion: {prediction}**")
                    st.info(f"**Vocabulary Coverage: {coverage:.1%}**")
                
                with col2:
                    st.subheader("📊 Confidence Scores")
                    if probabilities is not None:
                        prob_df = pd.DataFrame({
                            'Emotion': classifier.classes_,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        # Bar chart
                        fig = px.bar(
                            prob_df, 
                            x='Probability', 
                            y='Emotion',
                            orientation='h',
                            title="Emotion Probabilities"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top 3 predictions
                        st.subheader("🥇 Top 3 Predictions")
                        for i, row in prob_df.head(3).iterrows():
                            st.write(f"**{row['Emotion']}**: {row['Probability']:.3f}")
            else:
                st.warning("Please enter some lyrics to analyze.")
    
    with tab2:
        st.header("Batch Lyrics Classification")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with lyrics",
            type=['csv'],
            help="CSV should have a 'lyrics' column"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for lyrics column
                if 'lyrics' not in df.columns:
                    st.error("CSV file must contain a 'lyrics' column")
                    st.stop()
                
                # Process button
                if st.button("🚀 Process All Lyrics", type="primary"):
                    with st.spinner("Processing lyrics..."):
                        predictions = []
                        confidences = []
                        coverages = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, lyrics in enumerate(df['lyrics']):
                            if pd.isna(lyrics) or lyrics.strip() == '':
                                predictions.append('unknown')
                                confidences.append(0.0)
                                coverages.append(0.0)
                            else:
                                pred, probs, coverage = predict_emotion(
                                    lyrics, glove_embeddings, tfidf_vectorizer, scaler, classifier
                                )
                                predictions.append(pred)
                                confidences.append(max(probs) if probs is not None else 0.0)
                                coverages.append(coverage)
                            
                            # Update progress
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {idx + 1}/{len(df)} lyrics...")
                        
                        # Add results to dataframe
                        df['predicted_emotion'] = predictions
                        df['confidence'] = confidences
                        df['vocabulary_coverage'] = coverages
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success("✅ Processing completed!")
                    
                    # Results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Emotion Distribution")
                        emotion_counts = pd.Series(predictions).value_counts()
                        fig_pie = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            title="Distribution of Predicted Emotions"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Statistics")
                        st.metric("Total Processed", len(df))
                        st.metric("Average Confidence", f"{np.mean(confidences):.3f}")
                        st.metric("Average Coverage", f"{np.mean(coverages):.1%}")
                        
                        # Most confident predictions
                        st.subheader("🎯 Most Confident")
                        confident_df = df.nlargest(3, 'confidence')[['predicted_emotion', 'confidence']]
                        for _, row in confident_df.iterrows():
                            st.write(f"**{row['predicted_emotion']}**: {row['confidence']:.3f}")
                    
                    # Show results table
                    st.subheader("🗂️ Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="emotion_classification_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("👆 Please upload a CSV file to get started")
            
            # Show example format
            st.subheader("📋 Expected CSV Format")
            example_df = pd.DataFrame({
                'title': ['Song 1', 'Song 2', 'Song 3'],
                'artist': ['Artist A', 'Artist B', 'Artist C'],
                'lyrics': [
                    'I feel so happy today, dancing in the sunshine',
                    'Tears falling down, heart broken and alone',
                    'Burning with rage, cannot control my anger'
                ]
            })
            st.dataframe(example_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built using Streamlit | Powered by GloVe + TF-IDF")

if __name__ == "__main__":
    main()