const refreshForm = document.getElementById("refresh-data-form");
const refreshButton = document.getElementById("refresh-data-button");

refreshForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  refreshButton.disabled = true;
  refreshButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Refreshing...';
  fetch(refreshForm.action, {
    method: "POST",
    headers: {Accept: "application/json"},
  })
    .then((response) => {
      if (!response.ok) throw new Error("Could not refresh server data.");
      return response.json();
    })
    .then(() => waitForRefresh())
    .catch((error) => {
      refreshButton.disabled = false;
      refreshButton.textContent = "Refresh all data";
      refreshButton.title = error.message;
    });
});

function waitForRefresh() {
  fetch("/api/datasets/status")
    .then((response) => response.json())
    .then((state) => {
      if (state.status === "loading" || state.status === "pending") {
        window.setTimeout(waitForRefresh, 500);
      } else if (state.status === "ready") {
        window.location.reload();
      } else {
        throw new Error(state.error || "Could not refresh server data.");
      }
    })
    .catch((error) => {
      refreshButton.disabled = false;
      refreshButton.textContent = "Refresh all data";
      refreshButton.title = error.message;
    });
}

const processModal = document.getElementById("process-modal");
const processForm = document.getElementById("preprocess-form");
const processLabel = document.getElementById("process-source-label");
const recommendationStatus = document.getElementById("recommendation-status");
let activeProcessButton = null;
let activeProcessRow = null;

document.querySelectorAll(".process-source").forEach((button) => {
  button.addEventListener("click", () => {
    activeProcessButton = button;
    activeProcessRow = button.closest(".d-flex");
    processForm.dataset.dataset = button.dataset.dataset;
    processForm.dataset.source = button.dataset.source;
    document.getElementById("dataset_id").value = button.dataset.dataset;
    document.getElementById("dataset_source").value = button.dataset.source;
    processLabel.textContent = `${button.dataset.dataset} / ${button.dataset.source}`;
    processModal.showModal();
    recommendationStatus.textContent = "Inspecting source to suggest values...";
    const sourcePath = button.dataset.source.split("/").map(encodeURIComponent).join("/");
    fetch(`/datasets/${encodeURIComponent(button.dataset.dataset)}/sources/${sourcePath}/recommendations`)
      .then((response) => response.json().then((data) => ({ok: response.ok, data})))
      .then(({ok, data}) => {
        if (!ok) throw new Error(data.error || "Could not inspect source.");
        document.getElementById("target_spacing").value = data.target_spacing;
        document.getElementById("hu_range").value = data.intensity_range;
        recommendationStatus.textContent = `Suggested from input: shape ${data.input_shape.join(" × ")} and spacing ${data.input_spacing.join(" × ")}. You can adjust these values.`;
      })
      .catch((error) => {
        recommendationStatus.textContent = `Could not calculate suggestions: ${error.message}`;
      });
  });
});

processForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const button = activeProcessButton;
  const row = activeProcessRow;
  if (!button || !row) return;
  const status = row.querySelector(".processing-status");
  const preview = row.querySelector(".preview-source");
  const formData = new FormData(processForm);
  button.disabled = true;
  preview.classList.add("d-none");
  processModal.close();
  status.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status" aria-hidden="true"></span>Processing...';
  fetch(processForm.dataset.processUrl, {method: "POST", body: formData})
    .then((response) => response.json().then((data) => ({ok: response.ok, data})))
    .then(({ok, data}) => {
      if (!ok) throw new Error(data.error || "Could not start preprocessing.");
      pollProcessing(data.job_id, button, status, preview);
    })
    .catch((error) => {
      button.disabled = false;
      status.textContent = error.message;
    });
});

document.getElementById("process-cancel").addEventListener("click", () => processModal.close());

function pollProcessing(jobId, button, status, preview) {
  fetch(`/process/${jobId}/status`)
    .then((response) => response.json())
    .then((state) => {
      if (state.status === "queued" || state.status === "processing") {
        window.setTimeout(() => pollProcessing(jobId, button, status, preview), 1000);
      } else if (state.status === "complete") {
        button.disabled = false;
        status.textContent = "";
        preview.href = `/process/${jobId}/result`;
        preview.classList.remove("btn-outline-primary", "d-none");
        preview.classList.add("btn-success");
      } else {
        button.disabled = false;
        status.textContent = state.error || "Processing failed.";
      }
    })
    .catch(() => {
      button.disabled = false;
      status.textContent = "Unable to check processing status.";
    });
}

document.querySelectorAll(".process-source[data-job-id]").forEach((button) => {
  const row = button.closest(".d-flex");
  pollProcessing(
    button.dataset.jobId,
    button,
    row.querySelector(".processing-status"),
    row.querySelector(".preview-source")
  );
});

const deleteModal = document.getElementById("delete-modal");
const deleteForm = document.getElementById("delete-confirm-form");
const deleteLabel = document.getElementById("delete-label");
const deleteSource = document.getElementById("delete-source");

document.querySelectorAll(".delete-trigger").forEach((button) => {
  button.addEventListener("click", () => {
    deleteLabel.textContent = button.dataset.deleteLabel;
    deleteForm.action = button.dataset.deleteAction;
    deleteSource.value = button.dataset.deleteSource || "";
    deleteModal.showModal();
  });
});

document.getElementById("delete-cancel").addEventListener("click", () => deleteModal.close());
deleteModal.addEventListener("click", (event) => {
  if (event.target === deleteModal) deleteModal.close();
});
