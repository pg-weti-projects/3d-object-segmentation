from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    redirect,
    render_template,
    request,
    url_for,
)

from ..meshes import (
    MeshNotFound,
    build_mesh_from_mask,
    list_meshes,
    mesh_bytes,
    mesh_info,
)
from ..processing import process_uploads


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


@web.get("/viewer")
def viewer_page():
    return _render_viewer(request.args.get("id"))


@web.post("/viewer/build")
def viewer_build():
    upload = request.files.get("mask")
    if upload is None or not upload.filename:
        return _render_viewer(None, error="Choose a mask file."), 400
    try:
        mesh_id = build_mesh_from_mask(upload, request.form, _results_dir())
    except (OSError, ValueError) as error:
        current_app.logger.warning("viewer/build rejected %r: %s", upload.filename, error)
        return _render_viewer(None, error=f"Could not build the mesh: {error}"), 400
    return redirect(url_for("web.viewer_page", id=mesh_id))


@web.get("/meshes/<mesh_id>/mesh.<fmt>")
def mesh_file(mesh_id: str, fmt: str):
    """Mesh of a result in glb/stl/obj/ply. With ?download=1 the browser saves it."""
    try:
        content, mimetype = mesh_bytes(_results_dir(), mesh_id, fmt)
    except (MeshNotFound, ValueError):
        abort(404)
    response = Response(content, mimetype=mimetype)
    if request.args.get("download"):
        response.headers["Content-Disposition"] = f'attachment; filename="{mesh_id}.{fmt}"'
    return response


@web.get("/health")
def health():
    return {"status": "ok"}


def _results_dir() -> Path:
    return Path(current_app.config["RESULTS_DIR"])


def _render_viewer(selected_id: str | None, error: str | None = None):
    info = None
    if selected_id:
        try:
            info = mesh_info(_results_dir(), selected_id)
        except MeshNotFound:
            return render_template(
                "viewer.html",
                meshes=list_meshes(_results_dir()),
                error=f"No mesh found for result '{selected_id}'.",
            ), 404
    return render_template(
        "viewer.html",
        meshes=list_meshes(_results_dir()),
        info=info,
        error=error,
    )
