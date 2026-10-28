from pathlib import Path

import numpy as np
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
from ..experiments import page_context
from ..processing import process_uploads
from ..datasets import (
    DatasetError,
    delete_dataset,
    delete_source,
    import_dataset,
    load_dataset,
    rename_dataset,
)


web = Blueprint("web", __name__)


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/preprocess")
def preprocess_page():
    return render_template(
        "preprocess.html",
        datasets=_dataset_snapshot()["datasets"],
        processed_jobs=_processing_jobs().completed_for_sources(),
        active_jobs=_processing_jobs().active_for_sources(),
    )


@web.get("/datasets")
def datasets_page():
    return redirect(url_for("web.preprocess_page"))


@web.get("/api/datasets/status")
def datasets_status():
    return _dataset_snapshot()


@web.post("/datasets/refresh")
def datasets_refresh():
    _dataset_catalog().refresh_async()
    if request.headers.get("Accept") == "application/json":
        return {"status": "loading"}, 202
    return redirect(url_for("web.datasets_page"))


@web.post("/datasets")
def datasets_upload():
    try:
        dataset = import_dataset(
            _raw_data_dir(),
            request.form.get("dataset_id", ""),
            [file for file in request.files.getlist("files") if file.filename],
            [file for file in request.files.getlist("folder") if file.filename],
        )
    except DatasetError as error:
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error=str(error),
        ), 400
    _dataset_catalog().refresh_async()
    return render_template(
        "preprocess.html",
        datasets=_dataset_snapshot()["datasets"],
        message=f"Dataset '{dataset['id']}' was saved on the server.",
    )


@web.post("/datasets/<dataset_id>/delete")
def datasets_delete(dataset_id: str):
    try:
        delete_dataset(_raw_data_dir(), dataset_id)
    except DatasetError as error:
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error=str(error),
        ), 400
    _dataset_catalog().refresh()
    return redirect(url_for("web.preprocess_page"))


@web.post("/datasets/<dataset_id>/rename")
def datasets_rename(dataset_id: str):
    try:
        rename_dataset(
            _raw_data_dir(),
            dataset_id,
            request.form.get("new_dataset_id", ""),
        )
    except DatasetError as error:
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error=str(error),
        ), 400
    _dataset_catalog().refresh_async()
    new_id = request.form["new_dataset_id"]
    return render_template(
        "preprocess.html",
        datasets=_dataset_snapshot()["datasets"],
        message=f"Dataset '{dataset_id}' was renamed to '{new_id}'.",
    )


@web.post("/datasets/<dataset_id>/sources/delete")
def datasets_source_delete(dataset_id: str):
    source = request.form.get("source", "")
    try:
        delete_source(_raw_data_dir(), dataset_id, source)
    except DatasetError as error:
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error=str(error),
        ), 400
    _dataset_catalog().refresh()
    return redirect(url_for("web.preprocess_page"))


@web.get("/datasets/<dataset_id>/sources/<path:source>/recommendations")
def datasets_source_recommendations(dataset_id: str, source: str):
    try:
        volume = load_dataset(_raw_data_dir(), dataset_id, source)
        values = np.asarray(volume.data, dtype=np.float32)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError("The source contains no finite intensity values.")
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
    except (OSError, ValueError, DatasetError) as error:
        return {"error": f"Could not inspect the source: {error}"}, 400

    return {
        "target_spacing": ",".join(_format_recommendation(value) for value in volume.spacing_mm),
        "intensity_range": ",".join(
            (_format_recommendation(np.floor(minimum)), _format_recommendation(np.ceil(maximum)))
        ),
        "input_shape": list(values.shape),
        "input_spacing": list(volume.spacing_mm),
    }


@web.post("/process")
def process():
    uploads = [file for file in request.files.getlist("volume") if file.filename]
    dataset_id = request.form.get("dataset_id")
    dataset_source = request.form.get("dataset_source")
    if not uploads and not (dataset_id and dataset_source):
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error="Choose a server dataset or input file.",
        ), 400

    try:
        if uploads:
            result = process_uploads(uploads, request.form)
        else:
            volume = load_dataset(_raw_data_dir(), dataset_id, dataset_source)
            from ..processing import process_volume
            result = process_volume(volume, request.form)
    except (OSError, ValueError, DatasetError) as error:
        return render_template(
            "preprocess.html",
            datasets=_dataset_snapshot()["datasets"],
            error=f"Could not process the input: {error}",
        ), 400

    return render_template(
        "result.html",
        result=result,
        dataset_name=result.get("dataset_id"),
    )


@web.post("/process/start")
def process_start():
    dataset_id = request.form.get("dataset_id")
    dataset_source = request.form.get("dataset_source")
    if not dataset_id or not dataset_source:
        return {"error": "Select a validated server dataset source."}, 400
    form = request.form.to_dict()
    job_id = current_app.config["PROCESSING_JOBS"].start(
        _raw_data_dir(), dataset_id, dataset_source, form
    )
    return {"job_id": job_id}, 202


@web.get("/process/<job_id>/status")
def process_status(job_id: str):
    try:
        return current_app.config["PROCESSING_JOBS"].status(job_id)
    except KeyError:
        abort(404)


@web.get("/process/<job_id>/result")
def process_result(job_id: str):
    try:
        result = current_app.config["PROCESSING_JOBS"].result(job_id)
        source_info = _processing_jobs().source_info(job_id)
    except KeyError:
        abort(404)
    except ValueError:
        return {"error": "Processing has not completed."}, 409
    return render_template(
        "result.html",
        result=result,
        dataset_name=source_info["dataset_id"],
        preview_url=url_for("web.process_preview", job_id=job_id),
    )


@web.get("/process/<job_id>/preview")
def process_preview(job_id: str):
    try:
        plane = request.args.get("plane", "axial")
        index = int(request.args.get("index", "0"))
        data = _processing_jobs().preview_png(job_id, plane, index)
    except (KeyError, ValueError):
        abort(404)
    return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})


@web.get("/experiments")
def experiments_page():
    """Metrics of every run in results/, with a comparison chart."""
    return render_template("experiments.html", **page_context(_results_dir(), request.args.get("metric")))


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


def _raw_data_dir() -> Path:
    return Path(current_app.config["RAW_DATA_DIR"])


def _dataset_catalog():
    return current_app.config["DATASET_CATALOG"]


def _processing_jobs():
    return current_app.config["PROCESSING_JOBS"]


def _format_recommendation(value: float) -> str:
    return f"{value:.6g}"


def _dataset_snapshot():
    return _dataset_catalog().snapshot()


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
