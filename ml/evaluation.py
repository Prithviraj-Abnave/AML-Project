"""
Evaluation Module
=================
Calculates topic coherence, diversity, and other evaluation metrics.
"""

import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


def calculate_coherence(topics, processed_texts, coherence_type='c_v'):
    """
    Calculate topic coherence using gensim.

    Parameters
    ----------
    topics : list of dict
        Topics with top_words from the model.
    processed_texts : list of str
        Preprocessed texts (space-separated tokens).
    coherence_type : str
        Type of coherence measure ('c_v', 'c_npmi', 'u_mass').

    Returns
    -------
    float
        Coherence score.
    """
    # Convert processed texts to tokenized format
    tokenized_texts = [text.split() for text in processed_texts]

    # Create gensim dictionary
    dictionary = Dictionary(tokenized_texts)

    # Extract topic word lists
    topic_words = []
    for topic in topics:
        words = [w['word'] for w in topic['top_words'][:10]]
        topic_words.append(words)

    try:
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        score = coherence_model.get_coherence()
        return round(float(score), 4)
    except Exception as e:
        print(f"  Warning: Coherence calculation failed: {e}")
        return None


def calculate_diversity(topics, n_top_words=10):
    """
    Calculate topic diversity — the proportion of unique words
    in the top-N words across all topics.

    Higher diversity means topics are more distinct.

    Parameters
    ----------
    topics : list of dict
        Topics with top_words.
    n_top_words : int
        Number of top words to consider per topic.

    Returns
    -------
    float
        Diversity score between 0 and 1.
    """
    all_words = []
    for topic in topics:
        words = [w['word'] for w in topic['top_words'][:n_top_words]]
        all_words.extend(words)

    if len(all_words) == 0:
        return 0.0

    unique_words = set(all_words)
    diversity = len(unique_words) / len(all_words)

    return round(diversity, 4)


def calculate_all_metrics(model_name, topics, processed_texts,
                          model=None, matrix=None):
    """
    Calculate all available metrics for a model.

    Parameters
    ----------
    model_name : str
        'LDA' or 'NMF'.
    topics : list of dict
        Extracted topics.
    processed_texts : list of str
        Preprocessed texts.
    model : sklearn model, optional
        The trained model (for perplexity / reconstruction error).
    matrix : sparse matrix, optional
        The input matrix (for perplexity calculation).

    Returns
    -------
    dict
        Dictionary of metric_name: value.
    """
    metrics = {}

    # Coherence
    print(f"  Calculating coherence for {model_name}...")
    coherence = calculate_coherence(topics, processed_texts)
    if coherence is not None:
        metrics['coherence'] = coherence

    # Diversity
    metrics['diversity'] = calculate_diversity(topics)

    # Model-specific metrics
    if model_name == 'LDA' and model is not None and matrix is not None:
        try:
            from ml.lda_model import get_perplexity
            metrics['perplexity'] = get_perplexity(model, matrix)
        except Exception as e:
            print(f"  Warning: Perplexity calculation failed: {e}")

    if model_name == 'NMF' and model is not None:
        try:
            from ml.nmf_model import get_reconstruction_error
            metrics['reconstruction_error'] = get_reconstruction_error(model)
        except Exception as e:
            print(f"  Warning: Reconstruction error calculation failed: {e}")

    return metrics
