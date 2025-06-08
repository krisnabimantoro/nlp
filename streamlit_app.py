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
    glove_files = [
        'glove.6B.300d.txt',
        'glove.6B.200d.txt', 
        'glove.6B.100d.txt',
        'glove.6B.50d.txt'
    ]
    
    for glove_file in glove_files:
        try:
            embeddings_dict = {}
            embedding_dim = None
            with open(glove_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    if embedding_dim is None:
                        embedding_dim = len(vector)
                    embeddings_dict[word] = vector
            return embeddings_dict, embedding_dim, glove_file
        except FileNotFoundError:
            continue
    
    st.warning("No GloVe embeddings file found. GloVe option will be disabled.")
    return None, None, None

# Load models
@st.cache_resource
def load_models():
    """Load trained models and preprocessors"""
    models = {}
    
    # Try to load TF-IDF Decision Tree models
    try:
        models['tfidf_dt_classifier'] = joblib.load('tfidf_decision_tree_classifier.pkl')
        models['tfidf_vectorizer'] = joblib.load('tfidf_vectorizer.pkl')
        models['tfidf_dt_available'] = True
    except FileNotFoundError:
        models['tfidf_dt_available'] = False
    
    # Try to load TF-IDF Naive Bayes models
    try:
        models['tfidf_nb_classifier'] = joblib.load('tfidf_naive_bayes_classifier.pkl')
        models['tfidf_nb_available'] = True
    except FileNotFoundError:
        models['tfidf_nb_available'] = False
    
    # Try to load GloVe Gaussian Naive Bayes models
    try:
        models['glove_gaussian_nb_classifier'] = joblib.load('glove_gaussian_nb_classifier.pkl')
        models['glove_scaler'] = joblib.load('glove_scaler.pkl')
        models['glove_preprocessing_info'] = joblib.load('glove_preprocessing_info.pkl')
        models['glove_gaussian_nb_available'] = True
    except FileNotFoundError:
        models['glove_gaussian_nb_available'] = False
    
    # Try to load GloVe Decision Tree models (RESTORED FROM OLD VERSION)
    try:
        models['glove_dt_classifier'] = joblib.load('glove_emotion_classifier.pkl')
        models['glove_dt_scaler'] = joblib.load('glove_feature_scaler.pkl')
        models['glove_dt_tfidf'] = joblib.load('glove_tfidf_vectorizer.pkl')
        models['glove_dt_available'] = True
    except FileNotFoundError:
        models['glove_dt_available'] = False
    
    # Try to load Bag of Words Decision Tree models
    try:
        models['bow_dt_classifier'] = joblib.load('bow_decision_tree_classifier.pkl')
        models['bow_vectorizer'] = joblib.load('bow_vectorizer.pkl')
        models['bow_dt_available'] = True
    except FileNotFoundError:
        models['bow_dt_available'] = False
    
    # Try to load Bag of Words Naive Bayes models
    try:
        models['bow_nb_classifier'] = joblib.load('bow_naive_bayes_classifier.pkl')
        models['bow_nb_available'] = True
    except FileNotFoundError:
        models['bow_nb_available'] = False
    
    return models

# Text preprocessing functions
def preprocess_text(text):
    """Complete text preprocessing pipeline for TF-IDF and BoW"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' +', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def preprocess_text_for_glove(text):
    """Text preprocessing optimized for GloVe embeddings"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\'\-]', ' ', text)
    text = re.sub(' +', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words and len(word) > 1]
    
    return tokens

# GloVe embedding functions
def text_to_glove_vector(tokens, embeddings, embedding_dim):
    """Convert tokenized text to GloVe vector representation"""
    vectors = []
    found_words = 0
    
    for token in tokens:
        if token in embeddings:
            vectors.append(embeddings[token])
            found_words += 1
    
    if vectors:
        return np.mean(vectors, axis=0), found_words, len(tokens)
    else:
        return np.zeros(embedding_dim), 0, len(tokens)

def get_glove_embedding(tokens, embeddings_dict, embedding_dim=300):
    """Convert tokens to sentence embedding using GloVe (legacy function for compatibility)"""
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

# Prediction functions
def predict_emotion_tfidf(lyrics_text, vectorizer, classifier):
    """Predict emotion using TF-IDF"""
    processed_text = preprocess_text(lyrics_text)
    features = vectorizer.transform([processed_text]).toarray()
    
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0] if hasattr(classifier, 'predict_proba') else None
    
    return prediction, probabilities, {}

def predict_emotion_bow(lyrics_text, vectorizer, classifier):
    """Predict emotion using Bag of Words"""
    processed_text = preprocess_text(lyrics_text)
    features = vectorizer.transform([processed_text]).toarray()
    
    prediction = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0] if hasattr(classifier, 'predict_proba') else None
    
    return prediction, probabilities, {}

def predict_emotion_glove_gaussian_nb(lyrics_text, glove_dict, embedding_dim, scaler, classifier):
    """Predict emotion using GloVe + Gaussian Naive Bayes"""
    tokens = preprocess_text_for_glove(lyrics_text)
    
    vector, found_words, total_words = text_to_glove_vector(tokens, glove_dict, embedding_dim)
    
    scaled_features = scaler.transform([vector])
    
    prediction = classifier.predict(scaled_features)[0]
    probabilities = classifier.predict_proba(scaled_features)[0] if hasattr(classifier, 'predict_proba') else None
    
    coverage = found_words / total_words if total_words > 0 else 0
    
    return prediction, probabilities, {'coverage': coverage}

def predict_emotion_glove_dt(lyrics_text, glove_dict, tfidf_vectorizer, scaler, classifier, embedding_dim=300):
    """Predict emotion using GloVe Decision Tree (RESTORED FROM OLD VERSION)"""
    processed_text = preprocess_text(lyrics_text)
    tokens = processed_text.split()
    
    # GloVe embedding
    glove_features, found_words, total_words = get_glove_embedding(tokens, glove_dict, embedding_dim)
    
    # TF-IDF features
    tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray()[0]
    
    # Combine and predict
    combined_features = np.hstack([glove_features, tfidf_features])
    scaled_features = scaler.transform([combined_features])
    
    prediction = classifier.predict(scaled_features)[0]
    probabilities = classifier.predict_proba(scaled_features)[0] if hasattr(classifier, 'predict_proba') else None
    
    coverage = found_words / total_words if total_words > 0 else 0
    
    return prediction, probabilities, {'coverage': coverage}

# Get available models and representations
def get_available_models_and_representations(models):
    """Get available models with their supported text representations"""
    model_representations = {}
    
    # Decision Tree representations
    dt_reps = []
    if models.get('tfidf_dt_available', False):
        dt_reps.append("TF-IDF")
    if models.get('bow_dt_available', False):
        dt_reps.append("Bag of Words")
    if models.get('glove_dt_available', False):  # RESTORED
        dt_reps.append("GloVe")
    
    if dt_reps:
        model_representations["Decision Tree"] = dt_reps
    
    # Naive Bayes representations
    nb_reps = []
    if models.get('tfidf_nb_available', False):
        nb_reps.append("TF-IDF")
    if models.get('bow_nb_available', False):
        nb_reps.append("Bag of Words")
    if models.get('glove_gaussian_nb_available', False):
        nb_reps.append("GloVe")
    
    if nb_reps:
        model_representations["Naive Bayes"] = nb_reps
    
    return model_representations

# Get model information
def get_model_info(algorithm, representation):
    """Get detailed information about the selected model combination"""
    info = {
        'algorithm': algorithm,
        'representation': representation,
        'description': '',
        'advantages': [],
        'features': ''
    }
    
    # Algorithm-specific info
    if algorithm == "Decision Tree":
        info['advantages'] = [
            "Interpretable decision rules",
            "Handles non-linear relationships", 
            "No probabilistic assumptions",
            "Feature importance insights"
        ]
    elif algorithm == "Naive Bayes":
        info['advantages'] = [
            "Fast training and prediction",
            "Probabilistic output",
            "Good with small datasets",
            "Handles irrelevant features well"
        ]
    
    # Representation-specific info
    if representation == "TF-IDF":
        info['features'] = "Term Frequency-Inverse Document Frequency"
        info['description'] = "TF-IDF reduces impact of frequent words and highlights meaningful terms."
    elif representation == "Bag of Words":
        info['features'] = "Word frequency counts"
        info['description'] = "Simple word frequency counts with preserved frequency information."
    elif representation == "GloVe":
        info['features'] = "Word embeddings (300D)"
        info['description'] = "Semantic word embeddings that capture word relationships and context."
    elif representation == "GloVe":  # RESTORED
        info['features'] = "Word embeddings"
        info['description'] = "Combines semantic word embeddings with frequency-based features for better context understanding."
    
    return info

# Main prediction function
def make_prediction(lyrics_text, algorithm, representation, models, glove_embeddings, embedding_dim):
    """Make prediction based on selected algorithm and representation"""
    if algorithm == "Decision Tree":
        if representation == "TF-IDF":
            return predict_emotion_tfidf(
                lyrics_text, models['tfidf_vectorizer'], models['tfidf_dt_classifier']
            ), models['tfidf_dt_classifier']
        elif representation == "Bag of Words":
            return predict_emotion_bow(
                lyrics_text, models['bow_vectorizer'], models['bow_dt_classifier']
            ), models['bow_dt_classifier']
        elif representation == "GloVe":  # RESTORED
            return predict_emotion_glove_dt(
                lyrics_text, glove_embeddings, models['glove_dt_tfidf'], 
                models['glove_dt_scaler'], models['glove_dt_classifier'], embedding_dim
            ), models['glove_dt_classifier']
    
    elif algorithm == "Naive Bayes":
        if representation == "TF-IDF":
            return predict_emotion_tfidf(
                lyrics_text, models['tfidf_vectorizer'], models['tfidf_nb_classifier']
            ), models['tfidf_nb_classifier']
        elif representation == "Bag of Words":
            return predict_emotion_bow(
                lyrics_text, models['bow_vectorizer'], models['bow_nb_classifier']
            ), models['bow_nb_classifier']
        elif representation == "GloVe":
            return predict_emotion_glove_gaussian_nb(
                lyrics_text, glove_embeddings, embedding_dim, 
                models['glove_scaler'], models['glove_gaussian_nb_classifier']
            ), models['glove_gaussian_nb_classifier']
    
    raise ValueError(f"Unsupported combination: {algorithm} + {representation}")

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Emotion Classification Dashboard",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Lyrics Emotion Classification Dashboard")
    st.markdown("Classify emotions in song lyrics using different algorithms and text representations")
    
    # Initialize resources
    if not download_nltk_resources():
        st.warning("Some NLTK resources couldn't be downloaded. The app may not work properly.")
    
    # Load models and embeddings
    with st.spinner("Loading models and embeddings..."):
        glove_embeddings, embedding_dim, glove_file = load_glove_embeddings()
        models = load_models()
    
    # Get available models and representations
    model_representations = get_available_models_and_representations(models)
    
    if not model_representations:
        st.error("No models available. Please ensure model files are in the correct location.")
        return
    
    st.success(f"✅ Models loaded successfully!")
    if glove_embeddings is not None:
        st.info(f"GloVe loaded: {glove_file} ({embedding_dim}D)")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Select algorithm first
        available_algorithms = list(model_representations.keys())
        selected_algorithm = st.selectbox(
            "Choose Machine Learning Algorithm:",
            available_algorithms,
            help="Select the machine learning algorithm for classification"
        )
        
        # Select representation based on algorithm
        available_reps = model_representations[selected_algorithm]
        selected_representation = st.selectbox(
            "Choose Text Representation:",
            available_reps,
            help="Select how text will be converted to numerical features"
        )
        
        # Get model info
        model_info = get_model_info(selected_algorithm, selected_representation)
        
        st.header("📊 Model Information")
        st.write(f"**Algorithm:** {model_info['algorithm']}")
        st.write(f"**Representation:** {model_info['representation']}")
        st.write(f"**Features:** {model_info['features']}")
        
        if model_info['description']:
            st.info(model_info['description'])
        
        # Get classifier for additional info
        try:
            _, classifier = make_prediction("test", selected_algorithm, selected_representation, models, glove_embeddings, embedding_dim)
            st.write(f"**Classes:** {len(classifier.classes_)}")
            st.write(f"**Available Emotions:** {', '.join(classifier.classes_[:3])}{'...' if len(classifier.classes_) > 3 else ''}")
        except:
            pass
        
        # Algorithm advantages
        st.header("⚖️ Algorithm Advantages")
        st.write(f"**{selected_algorithm}:**")
        for advantage in model_info['advantages']:
            st.write(f"• {advantage}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔤 Single Prediction", "📁 Batch Prediction", "⚖️ Model Comparison"])
    
    with tab1:
        st.header("Single Lyrics Classification")
        
        # Show selected configuration
        st.info(f"Using: **{selected_algorithm} + {selected_representation}**")
        
        # Text input
        lyrics_input = st.text_area(
            "Enter song lyrics:",
            placeholder="Type or paste your song lyrics here...",
            height=150
        )
        
        if st.button("🎭 Predict Emotion", type="primary"):
            if lyrics_input.strip():
                with st.spinner("Analyzing lyrics..."):
                    try:
                        (prediction, probabilities, extra_info), classifier = make_prediction(
                            lyrics_input, selected_algorithm, selected_representation, 
                            models, glove_embeddings, embedding_dim
                        )
                        
                        # Results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Prediction Result")
                            st.success(f"**Predicted Emotion: {prediction.title()}**")
                            
                            # Show additional info if available
                            if 'coverage' in extra_info:
                                st.info(f"**Vocabulary Coverage: {extra_info['coverage']:.1%}**")
                            
                            # Show processed text preview
                            if selected_representation in ["GloVe", "GloVe"]:
                                if selected_representation == "GloVe":
                                    processed = ' '.join(preprocess_text_for_glove(lyrics_input))
                                else:
                                    processed = preprocess_text(lyrics_input)
                            else:
                                processed = preprocess_text(lyrics_input)
                            with st.expander("🔍 Processed Text Preview"):
                                st.text(processed[:200] + "..." if len(processed) > 200 else processed)
                        
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
                                    title=f"Emotion Probabilities",
                                    color='Probability',
                                    color_continuous_scale='viridis'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show top 3 predictions
                                st.subheader("🥇 Top 3 Predictions")
                                for i, row in prob_df.head(3).iterrows():
                                    confidence_emoji = "🎯" if row['Probability'] > 0.7 else "👍" if row['Probability'] > 0.5 else "🤔"
                                    st.write(f"{confidence_emoji} **{row['Emotion'].title()}**: {row['Probability']:.3f}")
                            else:
                                st.warning("Probability scores not available for this model.")
                    
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            else:
                st.warning("Please enter some lyrics to analyze.")
    
    with tab2:
        st.header("Batch Lyrics Classification")
        
        # Show selected configuration
        st.info(f"Using: **{selected_algorithm} + {selected_representation}**")
        
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
                        extra_infos = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, lyrics in enumerate(df['lyrics']):
                            if pd.isna(lyrics) or lyrics.strip() == '':
                                predictions.append('unknown')
                                confidences.append(0.0)
                                extra_infos.append({})
                            else:
                                try:
                                    (pred, probs, extra), classifier = make_prediction(
                                        lyrics, selected_algorithm, selected_representation, 
                                        models, glove_embeddings, embedding_dim
                                    )
                                    predictions.append(pred)
                                    confidences.append(max(probs) if probs is not None else 0.0)
                                    extra_infos.append(extra)
                                except Exception as e:
                                    predictions.append('error')
                                    confidences.append(0.0)
                                    extra_infos.append({})
                            
                            # Update progress
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {idx + 1}/{len(df)} lyrics...")
                        
                        # Add results to dataframe
                        df['predicted_emotion'] = predictions
                        df['confidence'] = confidences
                        
                        # Add coverage if available (GloVe models)
                        if selected_representation in ["GloVe", "GloVe"]:
                            coverages = [info.get('coverage', 0.0) for info in extra_infos]
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
                            title=f"Distribution ({selected_algorithm} + {selected_representation})"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Statistics")
                        st.metric("Total Processed", len(df))
                        st.metric("Average Confidence", f"{np.mean(confidences):.3f}")
                        
                        if selected_representation in ["GloVe", "GloVe"]:
                            coverages = [info.get('coverage', 0.0) for info in extra_infos]
                            st.metric("Average Coverage", f"{np.mean(coverages):.1%}")
                    
                    # Show results table
                    st.subheader("🗂️ Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"emotion_results_{selected_algorithm.lower().replace(' ', '_')}_{selected_representation.lower().replace(' ', '_').replace('+', '_')}.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("👆 Please upload a CSV file to get started")
    
    with tab3:
        st.header("Model Comparison")
        
        # Get all available combinations
        all_combinations = []
        for algorithm, representations in model_representations.items():
            for representation in representations:
                all_combinations.append((algorithm, representation))
        
        if len(all_combinations) > 1:
            comparison_text = st.text_area(
                "Enter lyrics to compare all available models:",
                placeholder="Type lyrics here to see how different models perform...",
                height=100,
                key="comparison_input"
            )
            
            if st.button("🔍 Compare All Models", type="secondary") and comparison_text.strip():
                with st.spinner("Running comparison across all models..."):
                    comparison_results = []
                    
                    for algorithm, representation in all_combinations:
                        try:
                            (pred, probs, extra), classifier = make_prediction(
                                comparison_text, algorithm, representation, 
                                models, glove_embeddings, embedding_dim
                            )
                            
                            max_confidence = max(probs) if probs is not None else 0.0
                            comparison_results.append({
                                'Algorithm': algorithm,
                                'Representation': representation,
                                'Model': f"{algorithm} + {representation}",
                                'Prediction': pred.title(),
                                'Confidence': max_confidence,
                                'Coverage': extra.get('coverage', None)
                            })
                        except Exception as e:
                            st.error(f"Error with {algorithm} + {representation}: {str(e)}")
                    
                    # Display comparison
                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Comparison Results")
                            display_df = comparison_df[['Model', 'Prediction', 'Confidence']]
                            if 'Coverage' in comparison_df.columns and comparison_df['Coverage'].notna().any():
                                display_df = comparison_df[['Model', 'Prediction', 'Confidence', 'Coverage']]
                            st.dataframe(display_df, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Confidence Comparison")
                            fig_comparison = px.bar(
                                comparison_df,
                                x='Model',
                                y='Confidence',
                                color='Prediction',
                                title="Model Confidence Comparison",
                                text='Confidence'
                            )
                            fig_comparison.update_xaxes(tickangle=45)
                            fig_comparison.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig_comparison.update_layout(height=500)
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Highlight best performing model
                        best_model = comparison_df.loc[comparison_df['Confidence'].idxmax()]
                        st.success(f"🏆 **Highest Confidence:** {best_model['Model']} with {best_model['Confidence']:.3f} confidence predicting '{best_model['Prediction']}'")
        else:
            st.info("Only one model combination available. Load more models to enable comparison.")
    
    # Footer
    st.markdown("---")
    total_combinations = sum(len(reps) for reps in model_representations.values())
    st.markdown(f"Built with Streamlit • Current: **{selected_algorithm} + {selected_representation}** • Available: **{total_combinations} model combinations**")

if __name__ == "__main__":
    main()