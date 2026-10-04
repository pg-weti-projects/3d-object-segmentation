from flask import Blueprint, render_template, request

from .processing import process_uploads


web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/preprocess")
def preprocess_page():
    return render_template("preprocess.html")


@web.get("/health")
def health():
    return {"status": "ok"}


@web.post("/process")
def process():
    uploads = [file for file in request.files.getlist("volume") if file.filename]
    if not uploads:
        return render_template("preprocess.html", error="Choose an input file."), 400

    try:
        result = process_uploads(uploads, request.form)
    except (OSError, ValueError) as error:
        return render_template(
            "preprocess.html", error=f"Could not process the input: {error}"
        ), 400

    return render_template("result.html", result=result)
