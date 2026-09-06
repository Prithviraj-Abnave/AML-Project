"""
NMF Model Module
================
Trains Non-negative Matrix Factorization model and extracts topic information.
"""

import time
import numpy as np
from sklearn.decomposition import NMF


def train_nmf(tfidf_matrix, n_topics=5, max_iter=300, random_state=42):
    """
    Train an NMF model.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF document-term matrix.
    n_topics : int
        Number of topics to extract.
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (nmf_model, training_time)
    """
    print(f"  Training NMF with {n_topics} topics...")
    start_time = time.time()

    nmf = NMF(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        init='nndsvd',
        solver='mu',
        beta_loss='frobenius'
    )
    nmf.fit(tfidf_matrix)

    training_time = time.time() - start_time
    print(f"  NMF training complete in {training_time:.2f}s")

    return nmf, training_time


def extract_topics(nmf_model, feature_names, n_top_words=15):
    """
    Extract topics with top words and their weights.

    Returns
    -------
    list of dict
        Each dict contains: id, top_words (list of {word, weight})
    """
    topics = []
    for idx, topic_dist in enumerate(nmf_model.components_):
        # Normalize
        total = topic_dist.sum()
        if total > 0:
            topic_dist_norm = topic_dist / total
        else:
            topic_dist_norm = topic_dist

        # Get top word indices
        top_indices = topic_dist_norm.argsort()[:-n_top_words - 1:-1]

        top_words = [
            {
                'word': feature_names[i],
                'weight': round(float(topic_dist_norm[i]), 6)
            }
            for i in top_indices
        ]

        topics.append({
            'id': idx,
            'top_words': top_words
        })

    return topics


def get_document_topics(nmf_model, tfidf_matrix):
    """
    Get topic distribution for each document.

    Returns
    -------
    np.ndarray
        Document-topic matrix of shape (n_docs, n_topics).
    """
    return nmf_model.transform(tfidf_matrix)


def get_topic_prevalence(doc_topic_matrix):
    """
    Calculate the prevalence of each topic across all documents.

    Returns
    -------
    list of float
        Prevalence for each topic.
    """
    dominant_topics = doc_topic_matrix.argmax(axis=1)
    n_topics = doc_topic_matrix.shape[1]
    n_docs = len(dominant_topics)

    prevalence = []
    for t in range(n_topics):
        count = int(np.sum(dominant_topics == t))
        prevalence.append(round(count / n_docs, 4))

    return prevalence


def get_representative_reviews(doc_topic_matrix, original_texts, n_reviews=3):
    """
    Get the most representative reviews for each topic.

    Returns
    -------
    dict
        {topic_id: [list of review texts]}
    """
    representatives = {}
    n_topics = doc_topic_matrix.shape[1]

    for t in range(n_topics):
        topic_scores = doc_topic_matrix[:, t]
        top_doc_indices = topic_scores.argsort()[-n_reviews:][::-1]

        reviews = []
        for idx in top_doc_indices:
            if idx < len(original_texts):
                review_text = original_texts[idx]
                if len(review_text) > 300:
                    review_text = review_text[:300] + '...'
                reviews.append(review_text)

        representatives[t] = reviews

    return representatives


def get_reconstruction_error(nmf_model):
    """Get the reconstruction error from the NMF model."""
    return round(float(nmf_model.reconstruction_err_), 2)
