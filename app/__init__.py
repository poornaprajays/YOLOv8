"""
Flask Application Factory
--------------------------
Creates and configures the Flask app instance.
"""

import os
from flask import Flask


def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Ensure upload folder exists
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    app.config["UPLOAD_FOLDER"] = upload_dir
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload
    app.config["SECRET_KEY"] = "yolov8-explorer-secret"

    # Register routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
