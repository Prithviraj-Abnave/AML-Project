"""
Topic Modeling Dashboard — Flask Application
=============================================
Compares LDA vs NMF topic models on Amazon product reviews.
"""

import os
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


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
