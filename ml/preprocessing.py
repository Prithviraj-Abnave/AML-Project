"""
Text Preprocessing Module
=========================
Handles all text cleaning, tokenization, lemmatization, and vectorization
for the topic modeling pipeline.
"""

import re
import time
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Custom stopwords for product reviews domain
CUSTOM_STOPWORDS = {
    'amazon', 'product', 'item', 'bought', 'buy', 'purchase', 'purchased',
    'order', 'ordered', 'would', 'could', 'also', 'really', 'get', 'got',
    'one', 'two', 'use', 'used', 'using', 'like', 'thing', 'things',
    'much', 'well', 'even', 'still', 'back', 'made', 'make', 'want',
    'say', 'said', 'need', 'know', 'way', 'lot', 'came', 'come',
    'going', 'went', 'take', 'took', 'put', 'give', 'gave', 'many',
    'great', 'good', 'nice', 'love', 'best', 'better', 'perfect',
    'excellent', 'amazing', 'wonderful', 'awesome', 'fantastic',
    'terrible', 'horrible', 'awful', 'worst', 'bad', 'poor',
    'review', 'reviews', 'star', 'stars', 'rating',
}


def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text):
    """Full preprocessing pipeline for a single text string."""
    # Clean
    text = clean_text(text)

    if not text:
        return ''

    # Tokenize
    tokens = word_tokenize(text)

    # Get stopwords
    stop_words = set(stopwords.words('english')).union(CUSTOM_STOPWORDS)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize, remove stopwords, and filter short tokens
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) >= 3
    ]

    return ' '.join(processed_tokens)


def preprocess_dataframe(df, text_column='reviews.text', min_length=10):
    """
    Preprocess a DataFrame of reviews.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing reviews.
    text_column : str
        Name of the column containing review text.
    min_length : int
        Minimum character length for a review to be included.

    Returns
    -------
    tuple
        (processed_texts, original_texts, valid_indices)
    """
    print(f"  Preprocessing {len(df)} reviews...")
    start_time = time.time()

    # Drop rows with missing text
    valid_mask = df[text_column].notna() & (df[text_column].str.len() >= min_length)
    df_valid = df[valid_mask].copy()

    original_texts = df_valid[text_column].tolist()

    # Apply preprocessing
    processed_texts = []
    for i, text in enumerate(original_texts):
        processed = preprocess_text(text)
        processed_texts.append(processed)
        if (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(original_texts)} reviews...")

    # Filter out empty processed texts
    valid_indices = []
    filtered_processed = []
    filtered_original = []

    for i, (proc, orig) in enumerate(zip(processed_texts, original_texts)):
        if len(proc.split()) >= 3:  # At least 3 words after processing
            valid_indices.append(i)
            filtered_processed.append(proc)
            filtered_original.append(orig)

    elapsed = time.time() - start_time
    print(f"  Preprocessing complete: {len(filtered_processed)} valid reviews "
          f"(removed {len(original_texts) - len(filtered_processed)} short/empty) "
          f"in {elapsed:.1f}s")

    return filtered_processed, filtered_original, valid_indices


def _fit_vectorizer_safely(vectorizer_cls, processed_texts, max_features=5000, min_df=5, max_df=0.85):
    """Safely fit CountVectorizer or TfidfVectorizer, adapting df parameters for small corpora."""
    n_docs = len(processed_texts)
    
    # Calculate sensible initial min_df / max_df based on corpus size
    if n_docs < 15:
        initial_min_df, initial_max_df = 1, 1.0
    elif n_docs < 30:
        initial_min_df, initial_max_df = min(2, min_df), min(0.95, max_df)
    elif n_docs < 100:
        initial_min_df, initial_max_df = min(3, min_df), min(0.90, max_df)
    else:
        initial_min_df, initial_max_df = min_df, max_df

    candidates = [
        (initial_min_df, initial_max_df),
        (2, 0.95),
        (1, 1.0)
    ]

    for m_df, mx_df in candidates:
        try:
            vec = vectorizer_cls(
                max_features=max_features,
                min_df=m_df,
                max_df=mx_df,
                ngram_range=(1, 2)
            )
            matrix = vec.fit_transform(processed_texts)
            feature_names = vec.get_feature_names_out()
            if len(feature_names) > 0:
                return matrix, vec, feature_names
        except ValueError:
            continue

    # Fallback: unigrams without pruning constraints
    vec = vectorizer_cls(max_features=max_features, min_df=1, max_df=1.0, ngram_range=(1, 1))
    matrix = vec.fit_transform(processed_texts)
    feature_names = vec.get_feature_names_out()
    return matrix, vec, feature_names


def create_bow_matrix(processed_texts, max_features=5000, min_df=5, max_df=0.85):
    """
    Create Bag-of-Words (Count) matrix for LDA.

    Returns
    -------
    tuple
        (bow_matrix, count_vectorizer, feature_names)
    """
    print("  Creating Bag-of-Words matrix...")
    bow_matrix, vectorizer, feature_names = _fit_vectorizer_safely(
        CountVectorizer, processed_texts, max_features=max_features, min_df=min_df, max_df=max_df
    )
    print(f"  BoW matrix shape: {bow_matrix.shape}, vocabulary size: {len(feature_names)}")
    return bow_matrix, vectorizer, feature_names


def create_tfidf_matrix(processed_texts, max_features=5000, min_df=5, max_df=0.85):
    """
    Create TF-IDF matrix for NMF.

    Returns
    -------
    tuple
        (tfidf_matrix, tfidf_vectorizer, feature_names)
    """
    print("  Creating TF-IDF matrix...")
    tfidf_matrix, vectorizer, feature_names = _fit_vectorizer_safely(
        TfidfVectorizer, processed_texts, max_features=max_features, min_df=min_df, max_df=max_df
    )
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}, vocabulary size: {len(feature_names)}")
    return tfidf_matrix, vectorizer, feature_names
