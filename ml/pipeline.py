"""
Pipeline Orchestrator
=====================
Central module that ties together the entire topic modeling pipeline:
load data → preprocess → train LDA + NMF → evaluate → compare → save results.
"""

import os
import json
import time
import threading
import numpy as np
import pandas as pd

from ml.preprocessing import preprocess_dataframe, create_bow_matrix, create_tfidf_matrix
from ml.lda_model import (
    train_lda, extract_topics as lda_extract_topics,
    get_document_topics as lda_doc_topics,
    get_topic_prevalence as lda_prevalence,
    get_representative_reviews as lda_representatives,
    get_perplexity
)
from ml.nmf_model import (
    train_nmf, extract_topics as nmf_extract_topics,
    get_document_topics as nmf_doc_topics,
    get_topic_prevalence as nmf_prevalence,
    get_representative_reviews as nmf_representatives,
    get_reconstruction_error
)
from ml.evaluation import calculate_all_metrics
from ml.topic_similarity import compute_topic_similarity


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Global pipeline state
_pipeline_state = {
    'status': 'idle',       # idle | running | complete | error
    'progress': 0,          # 0-100
    'current_step': '',
    'results': None,
    'error': None,
    'is_custom': False,
    'lock': threading.Lock()
}


def get_status():
    """Get current pipeline status."""
    with _pipeline_state['lock']:
        return {
            'status': _pipeline_state['status'],
            'progress': _pipeline_state['progress'],
            'current_step': _pipeline_state['current_step'],
            'has_results': _pipeline_state['results'] is not None,
            'error': _pipeline_state['error'],
            'is_custom': _pipeline_state['is_custom']
        }


def get_results():
    """Get cached results."""
    with _pipeline_state['lock']:
        return _pipeline_state['results']


def _update_state(status=None, progress=None, step=None, results=None, error=None):
    """Thread-safe state update."""
    with _pipeline_state['lock']:
        if status is not None:
            _pipeline_state['status'] = status
        if progress is not None:
            _pipeline_state['progress'] = progress
        if step is not None:
            _pipeline_state['current_step'] = step
            print(f"[Pipeline] {step}")
        if results is not None:
            _pipeline_state['results'] = results
        if error is not None:
            _pipeline_state['error'] = error

def _set_is_custom(is_custom):
    with _pipeline_state['lock']:
        _pipeline_state['is_custom'] = is_custom


def run_pipeline(n_topics=5, sample_size=None, data_path=None, is_custom=False):
    """
    Run the full topic modeling pipeline in a background thread.

    Parameters
    ----------
    n_topics : int
        Number of topics for both LDA and NMF.
    sample_size : int or None
        Number of reviews to sample (None = use all).
    data_path : str or None
        Path to the CSV file.
    is_custom : bool
        Whether this is a user-uploaded custom dataset.
    """
    _set_is_custom(is_custom)
    thread = threading.Thread(
        target=_run_pipeline_worker,
        args=(n_topics, sample_size, data_path, is_custom),
        daemon=True
    )
    thread.start()


def _run_pipeline_worker(n_topics, sample_size, data_path, is_custom):
    """Worker function that runs in a background thread."""
    try:
        _update_state(status='running', progress=0, step='Starting pipeline...', error=None)
        pipeline_start = time.time()

        # ── Step 1: Load Data ───────────────────────────────────────────
        _update_state(progress=5, step='Loading data...')

        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'data', 'amazon_reviews.csv')

        df = pd.read_csv(data_path)
        total_reviews = len(df)

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)

        _update_state(progress=10, step=f'Loaded {len(df)} reviews (total in dataset: {total_reviews})')

        # ── Step 2: Preprocess ──────────────────────────────────────────
        _update_state(progress=15, step='Preprocessing text...')
        
        if is_custom:
            text_column = None
            max_avg_len = -1
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Detect longest text column by average string length
                    avg_len = df[col].dropna().astype(str).str.len().mean()
                    if avg_len > max_avg_len:
                        max_avg_len = avg_len
                        text_column = col
            if not text_column:
                raise ValueError("Could not find a valid text column in the uploaded CSV.")
            text_col_name = text_column
        else:
            text_col_name = 'reviews.text'

        processed_texts, original_texts, valid_indices = preprocess_dataframe(
            df, text_column=text_col_name
        )
        _update_state(progress=30, step=f'Preprocessed {len(processed_texts)} valid reviews')

        if len(processed_texts) < n_topics:
            raise ValueError(f"Not enough valid reviews ({len(processed_texts)}) to extract {n_topics} topics. Try reducing the number of topics or scraping more data.")

        # ── Step 3: Create Matrices ─────────────────────────────────────
        _update_state(progress=35, step='Creating document-term matrices...')
        bow_matrix, bow_vectorizer, bow_features = create_bow_matrix(processed_texts)
        tfidf_matrix, tfidf_vectorizer, tfidf_features = create_tfidf_matrix(processed_texts)
        _update_state(progress=40, step='Matrices created')

        # ── Step 4: Train LDA ───────────────────────────────────────────
        _update_state(progress=42, step='Training LDA model...')
        lda_model, lda_time = train_lda(bow_matrix, n_topics=n_topics)
        lda_topics = lda_extract_topics(lda_model, bow_features)
        lda_doc_topic_matrix = lda_doc_topics(lda_model, bow_matrix)
        lda_prev = lda_prevalence(lda_doc_topic_matrix)
        lda_reps = lda_representatives(lda_doc_topic_matrix, original_texts)
        _update_state(progress=55, step=f'LDA trained in {lda_time:.1f}s')

        # ── Step 5: Train NMF ───────────────────────────────────────────
        _update_state(progress=57, step='Training NMF model...')
        nmf_model, nmf_time = train_nmf(tfidf_matrix, n_topics=n_topics)
        nmf_topics = nmf_extract_topics(nmf_model, tfidf_features)
        nmf_doc_topic_matrix = nmf_doc_topics(nmf_model, tfidf_matrix)
        nmf_prev = nmf_prevalence(nmf_doc_topic_matrix)
        nmf_reps = nmf_representatives(nmf_doc_topic_matrix, original_texts)
        _update_state(progress=70, step=f'NMF trained in {nmf_time:.1f}s')

        # ── Step 6: Evaluate ────────────────────────────────────────────
        _update_state(progress=72, step='Evaluating models...')
        lda_metrics = calculate_all_metrics(
            'LDA', lda_topics, processed_texts,
            model=lda_model, matrix=bow_matrix
        )
        nmf_metrics = calculate_all_metrics(
            'NMF', nmf_topics, processed_texts,
            model=nmf_model, matrix=tfidf_matrix
        )
        _update_state(progress=85, step='Evaluation complete')

        # ── Step 7: Cross-Model Similarity ──────────────────────────────
        _update_state(progress=87, step='Computing topic similarity...')
        similarity_matrix = compute_topic_similarity(
            lda_model, nmf_model, bow_features, tfidf_features
        )
        _update_state(progress=92, step='Similarity computed')

        # ── Step 8: Package Results ─────────────────────────────────────
        _update_state(progress=95, step='Packaging results...')
        total_time = time.time() - pipeline_start

        results = {
            'meta': {
                'total_reviews': total_reviews,
                'processed_reviews': len(processed_texts),
                'sample_size': sample_size,
                'n_topics': n_topics,
                'total_time': round(total_time, 2),
                'vocabulary_size_bow': len(bow_features),
                'vocabulary_size_tfidf': len(tfidf_features),
            },
            'lda': {
                'topics': lda_topics,
                'prevalence': lda_prev,
                'representative_reviews': {str(k): v for k, v in lda_reps.items()},
                'metrics': lda_metrics,
                'training_time': round(lda_time, 2),
            },
            'nmf': {
                'topics': nmf_topics,
                'prevalence': nmf_prev,
                'representative_reviews': {str(k): v for k, v in nmf_reps.items()},
                'metrics': nmf_metrics,
                'training_time': round(nmf_time, 2),
            },
            'similarity': {
                'matrix': similarity_matrix,
                'lda_labels': [f'LDA Topic {i}' for i in range(n_topics)],
                'nmf_labels': [f'NMF Topic {i}' for i in range(n_topics)],
            }
        }

        # Save to file
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )
        os.makedirs(results_dir, exist_ok=True)
        filename = 'custom_results.json' if is_custom else 'results.json'
        results_path = os.path.join(results_dir, filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        _update_state(
            status='complete', progress=100,
            step=f'Pipeline complete in {total_time:.1f}s',
            results=results
        )

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Pipeline Error] {error_msg}")
        traceback.print_exc()
        _update_state(status='error', progress=0, step='Error occurred', error=error_msg)


def load_cached_results():
    """Load results from disk if available."""
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'results.json'
    )
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        _update_state(status='complete', progress=100,
                      step='Loaded cached results', results=results)
        return results
    return None

def load_custom_results():
    """Load custom dataset results from disk if available."""
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'custom_results.json'
    )
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None
