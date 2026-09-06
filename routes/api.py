"""
API Routes — Pipeline control and results retrieval
"""

import os
from werkzeug.utils import secure_filename
from flask import Blueprint, jsonify, request
from ml.pipeline import run_pipeline, get_status, get_results, load_cached_results, load_custom_results

api_bp = Blueprint('api', __name__)


@api_bp.route('/status')
def status():
    """Get current pipeline status."""
    return jsonify(get_status())


@api_bp.route('/results')
def results():
    """Get analysis results (from memory or disk cache)."""
    data = get_results()
    if data is None:
        # Try loading from disk
        data = load_cached_results()

    if data is None:
        return jsonify({'error': 'No results available. Run the analysis first.'}), 404

    return jsonify(data)


@api_bp.route('/custom_results')
def custom_results():
    """Get custom analysis results."""
    data = load_custom_results()
    if data is None:
        return jsonify({'error': 'No custom results available.'}), 404
    return jsonify(data)


@api_bp.route('/analyze', methods=['POST'])
def analyze():
    """Kick off the topic modeling pipeline."""
    current = get_status()
    if current['status'] == 'running':
        return jsonify({'error': 'Pipeline is already running.'}), 409

    # Parse parameters
    body = request.get_json(silent=True) or {}
    n_topics = int(body.get('n_topics', 5))
    sample_size = body.get('sample_size')

    if sample_size is not None:
        sample_size = int(sample_size)

    # Validate
    if n_topics < 2 or n_topics > 20:
        return jsonify({'error': 'n_topics must be between 2 and 20.'}), 400

    if sample_size is not None and sample_size < 100:
        return jsonify({'error': 'sample_size must be at least 100.'}), 400

    # Run pipeline in background thread
    run_pipeline(n_topics=n_topics, sample_size=sample_size)

    return jsonify({'message': 'Pipeline started.', 'n_topics': n_topics, 'sample_size': sample_size})


@api_bp.route('/upload', methods=['POST'])
def upload():
    """Handle custom CSV upload and start pipeline."""
    current = get_status()
    if current['status'] == 'running':
        return jsonify({'error': 'Pipeline is already running.'}), 409

    if 'file' not in request.files:
        return jsonify({'error': 'No file part.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed.'}), 400

    n_topics = int(request.form.get('n_topics', 5))
    sample_size = request.form.get('sample_size')
    if sample_size and sample_size != 'null':
        sample_size = int(sample_size)
    else:
        sample_size = None

    # Save file
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    filename = secure_filename(file.filename)
    filepath = os.path.join(uploads_dir, filename)
    file.save(filepath)

    # Run pipeline with custom dataset
    run_pipeline(n_topics=n_topics, sample_size=sample_size, data_path=filepath, is_custom=True)

    return jsonify({'message': 'File uploaded and pipeline started.'})
