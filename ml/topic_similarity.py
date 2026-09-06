"""
Topic Similarity Module
=======================
Computes cross-model topic similarity using cosine similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_topic_similarity(lda_model, nmf_model, feature_names_lda, feature_names_nmf):
    """
    Compute cosine similarity between LDA and NMF topics.

    Both models' topic-word matrices are aligned to a common vocabulary
    before computing similarity.

    Parameters
    ----------
    lda_model : sklearn LatentDirichletAllocation
        Trained LDA model.
    nmf_model : sklearn NMF
        Trained NMF model.
    feature_names_lda : array-like
        Feature names from CountVectorizer.
    feature_names_nmf : array-like
        Feature names from TfidfVectorizer.

    Returns
    -------
    list of list
        Cosine similarity matrix (LDA topics x NMF topics).
    """
    # Get common vocabulary
    vocab_lda = set(feature_names_lda)
    vocab_nmf = set(feature_names_nmf)
    common_vocab = sorted(vocab_lda.intersection(vocab_nmf))

    if len(common_vocab) == 0:
        print("  Warning: No common vocabulary between LDA and NMF!")
        return []

    # Build indices mapping
    lda_idx = {word: i for i, word in enumerate(feature_names_lda)}
    nmf_idx = {word: i for i, word in enumerate(feature_names_nmf)}

    # Extract and align topic-word distributions
    n_lda_topics = lda_model.components_.shape[0]
    n_nmf_topics = nmf_model.components_.shape[0]

    lda_aligned = np.zeros((n_lda_topics, len(common_vocab)))
    nmf_aligned = np.zeros((n_nmf_topics, len(common_vocab)))

    for j, word in enumerate(common_vocab):
        if word in lda_idx:
            lda_aligned[:, j] = lda_model.components_[:, lda_idx[word]]
        if word in nmf_idx:
            nmf_aligned[:, j] = nmf_model.components_[:, nmf_idx[word]]

    # Normalize rows
    for i in range(n_lda_topics):
        norm = np.linalg.norm(lda_aligned[i])
        if norm > 0:
            lda_aligned[i] /= norm

    for i in range(n_nmf_topics):
        norm = np.linalg.norm(nmf_aligned[i])
        if norm > 0:
            nmf_aligned[i] /= norm

    # Compute cosine similarity
    sim_matrix = cosine_similarity(lda_aligned, nmf_aligned)

    # Round for readability
    result = [[round(float(sim_matrix[i][j]), 3)
               for j in range(n_nmf_topics)]
              for i in range(n_lda_topics)]

    return result
