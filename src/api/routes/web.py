from flask import Blueprint, render_template


web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/health")
def health():
    return {"status": "ok"}
