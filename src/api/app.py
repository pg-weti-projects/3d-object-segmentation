"""Flask application entry point."""

from flask import Flask

from .config import load_settings
from .routes import web


def create_app() -> Flask:
    """Create the web application."""
    app = Flask(__name__)
    settings = load_settings()
    app.config.update(
        DEBUG=bool(settings.get("debug", False)),
        HOST=str(settings.get("host", "127.0.0.1")),
        PORT=int(settings.get("port", 5000)),
    )
    app.register_blueprint(web)
    return app

app = create_app()


if __name__ == "__main__":
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
    )
