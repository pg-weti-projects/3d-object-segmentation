from pathlib import Path

from flask import Blueprint, current_app, render_template, request
from pydicom.errors import InvalidDicomError

from ..processing import process_uploads
from ..segmentation import default_store_dir, form_context, load_uploads, reload_view, run_segmentation

web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/preprocess")
def preprocess_page():
    return render_template("preprocess.html")


@web.post("/process")
def process():
    uploads = [file for file in request.files.getlist("volume") if file.filename]
    if not uploads:
        return render_template(
            "preprocess.html", error="Choose an input file."
        ), 400

    try:
        result = process_uploads(uploads, request.form)
    except (OSError, ValueError) as error:
        return render_template(
            "preprocess.html",
            error=f"Could not process the input: {error}",
        ), 400

    return render_template("result.html", result=result)


@web.get("/segment")
def segment_page():
    return render_template("segment.html")


def _store_dir() -> Path:
    return Path(current_app.config.get("SEGMENTATION_STORE_DIR") or default_store_dir())


@web.post("/segment")
def segment():
    uploads = [file for file in request.files.getlist("volume") if file.filename]
    if not uploads:
        return render_template("segment.html", error="Choose an input file.", form=request.form), 400

    try:
        view = load_uploads(uploads, request.form, _store_dir())
    except (OSError, ValueError, InvalidDicomError) as error:
        return render_template(
            "segment.html", error=f"Could not load the input: {error}", form=request.form
        ), 400

    return render_template("segment_view.html", view=view, **form_context())


@web.post("/segment/run")
def segment_run():
    try:
        view = run_segmentation(request.form, _store_dir())
    except (ValueError, TypeError, RuntimeError) as error:
        try:
            view = reload_view(request.form, _store_dir())
        except ValueError as missing:
            return render_template("segment.html", error=str(missing)), 400
        return render_template(
            "segment_view.html",
            view=view,
            error=f"Could not segment the volume: {error}",
            form=request.form,
            **form_context(),
        ), 400

    return render_template("segment_view.html", view=view, form=request.form, **form_context())


@web.get("/health")
def health():
    return {"status": "ok"}
