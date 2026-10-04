"""Flask application entry point."""

from pathlib import Path
import tomllib

from flask import Flask, render_template


def create_app() -> Flask:
    """Create web app with config file."""
    app = Flask(__name__)
    config_path = Path(__file__).with_name("config.toml.example")
    settings = {}
    if config_path.exists():
        with config_path.open("rb") as config_file:
            settings = tomllib.load(config_file).get("app", {})
    app.config.update(
        DEBUG=bool(settings.get("debug", False)),
        HOST=str(settings.get("host", "127.0.0.1")),
        PORT=int(settings.get("port", 5000)),
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
    )
