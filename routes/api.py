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


@api_bp.route('/scrape', methods=['POST'])
def scrape():
    """
    Scrape reviews from product URLs and run topic modeling pipeline.

    Expected JSON body:
    {
        "urls": ["https://www.flipkart.com/...", "https://www.snapdeal.com/...", "..."],
        "n_topics": 5,
        "max_pages": 3
    }
    """
    current = get_status()
    if current['status'] == 'running':
        return jsonify({'error': 'Pipeline is already running.'}), 409

    body = request.get_json(silent=True) or {}
    urls = body.get('urls', [])
    n_topics = int(body.get('n_topics', 5))
    max_pages = int(body.get('max_pages', 3))

    # Validate
    urls = [u for u in urls if u and str(u).strip()]
    if len(urls) < 2:
        return jsonify({'error': 'Please provide at least 2 product URLs.'}), 400

    if len(urls) > 4:
        return jsonify({'error': 'Maximum 4 URLs are supported.'}), 400

    if n_topics < 2 or n_topics > 20:
        return jsonify({'error': 'n_topics must be between 2 and 20.'}), 400

    # Validate that URLs look plausible
    supported_hosts = ('flipkart.com', 'snapdeal.com', 'myntra.com', 'amazon.in', 'amazon.com')
    for url in urls:
        if not any(host in url.lower() for host in supported_hosts):
            return jsonify({
                'error': f'Unsupported site in URL: {url}. '
                         'Supported: Flipkart, Snapdeal, Myntra, Amazon.'
            }), 400

    # Kick off scraping + pipeline in a background thread
    _run_scrape_pipeline(urls, n_topics, max_pages)

    return jsonify({
        'message': 'Scraping started.',
        'urls': urls,
        'n_topics': n_topics
    })


def _run_scrape_pipeline(urls, n_topics, max_pages):
    """Launch scrape + pipeline in a background thread."""
    import threading
    from ml.pipeline import _update_state, _set_is_custom, _run_pipeline_worker
    from ml.scraper import scrape_reviews

    def worker():
        _set_is_custom(True)
        _update_state(status='running', progress=0,
                      step='Starting web scraping...', error=None)
        try:
            # ── Scrape reviews from all URLs ─────────────────────────────
            _update_state(progress=5, step='Scraping reviews from provided URLs...')
            df = scrape_reviews(urls, max_pages=max_pages)

            total = len(df)
            _update_state(
                progress=18,
                step=f'Scraped {total} reviews from {df["source"].nunique()} site(s). '
                     f'({", ".join(df["source"].value_counts().to_dict().keys())})'
            )

            # ── Save scraped data as a temp CSV ──────────────────────────
            uploads_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'uploads'
            )
            os.makedirs(uploads_dir, exist_ok=True)
            csv_path = os.path.join(uploads_dir, 'scraped_reviews.csv')
            df.to_csv(csv_path, index=False)

            _update_state(progress=20, step='Saved scraped data. Running topic model pipeline...')

            # ── Hand off to the existing pipeline worker ──────────────────
            # We bypass run_pipeline() (which would spawn a new thread) and call
            # the worker directly since we're already in a thread.
            _run_pipeline_worker(
                n_topics=n_topics,
                sample_size=None,
                data_path=csv_path,
                is_custom=True
            )

        except ValueError as e:
            _update_state(status='error', progress=0,
                          step='Scraping failed', error=str(e))
        except Exception as e:
            import traceback
            traceback.print_exc()
            _update_state(status='error', progress=0,
                          step='Unexpected error during scraping',
                          error=f"{type(e).__name__}: {e}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
