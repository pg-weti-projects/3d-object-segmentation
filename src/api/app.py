from pathlib import Path

from flask import Flask

from .config import load_settings
from .datasets import DatasetCatalog
from .processing_jobs import ProcessingJobStore
from .routes.web import web


def create_app() -> Flask:
    """Create the Flask application."""
    app = Flask(__name__)
    settings = load_settings()
    raw_data_dir = _repository_root() / str(settings.get("raw_data_dir", "data/raw"))
    app.config.update(
        DEBUG=bool(settings.get("debug", False)),
        HOST=str(settings.get("host", "127.0.0.1")),
        PORT=int(settings.get("port", 5000)),
        MAX_FORM_PARTS=int(settings.get("max_form_parts", 5000)),
        # where experiment results (results/<id>/mesh.*) live -> relative paths
        # are resolved from the repository root
        RESULTS_DIR=_repository_root() / str(settings.get("results_dir", "results")),
        RAW_DATA_DIR=raw_data_dir,
        DATASET_CATALOG=DatasetCatalog(raw_data_dir),
        PROCESSING_JOBS=ProcessingJobStore(
            _repository_root() / str(settings.get("results_dir", "results"))
        ),
    )
    app.config["DATASET_CATALOG"].start()
    app.register_blueprint(web)
    return app


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]



if __name__ == "__main__":
    app = create_app()

    app.run(
        host=app.config["HOST"],
        port=app.config["PORT"],
        debug=app.config["DEBUG"],
    )
