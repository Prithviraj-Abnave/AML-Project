"""
Topic Modeling Dashboard — Dual Streamlit / Flask Application
=============================================================
Compares LDA vs NMF topic models on Amazon product reviews.

Supports:
  1. streamlit run streamlit_app.py
  2. streamlit run app.py
  3. python app.py (Flask standalone)
"""

import os
import sys

# Check if launched via Streamlit CLI
_running_in_streamlit = False
try:
    import streamlit as st
    if st.runtime.exists():
        _running_in_streamlit = True
        import streamlit_app
        streamlit_app.main()
except Exception:
    pass

if not _running_in_streamlit:
    from flask import Flask
    from routes.main import main_bp
    from routes.api import api_bp

    def create_app():
        """Flask application factory."""
        app = Flask(
            __name__,
            static_folder='static',
            template_folder='templates'
        )

        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'topic-modeling-dev-key')

        # Ensure results directory exists
        os.makedirs(os.path.join(app.root_path, 'results'), exist_ok=True)

        # Register blueprints
        app.register_blueprint(main_bp)
        app.register_blueprint(api_bp, url_prefix='/api')

        return app

    # Expose WSGI app for gunicorn
    app = create_app()

    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        print(f"\n==================================================")
        print(f"  Topic Modeling Dashboard is running!")
        print(f"  Open in your browser: http://127.0.0.1:{port}")
        print(f"==================================================\n")
        app.run(debug=True, host='0.0.0.0', port=port)
