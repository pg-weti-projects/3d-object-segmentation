const fileInput = document.getElementById("volume");
const form = document.getElementById("preprocess-form");

async function monitorDatasetLoading() {
  const banner = document.getElementById("dataset-load-status");
  if (!banner) return;
  try {
    const response = await fetch("/api/datasets/status");
    const state = await response.json();
    if (state.status === "pending" || state.status === "loading") {
      banner.className = "alert alert-info rounded-0 mb-0";
      if (banner.dataset.loading !== "true") {
        banner.innerHTML = `
          <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
          <span>Loading server data...</span>
          <span class="visually-hidden">Please wait while the server data is indexed.</span>
        `;
        banner.dataset.loading = "true";
      }
      window.setTimeout(monitorDatasetLoading, 1000);
    } else if (state.status === "error") {
      delete banner.dataset.loading;
      banner.className = "alert alert-danger rounded-0 mb-0";
      banner.textContent = `Failed to load server data: ${state.error}`;
    } else {
      delete banner.dataset.loading;
      if (document.querySelector("[data-datasets-loading]")) {
        window.setTimeout(() => window.location.reload(), 300);
      } else {
        banner.className = "alert alert-info rounded-0 mb-0 d-none";
      }
    }
  } catch (_) {
    banner.className = "alert alert-warning rounded-0 mb-0";
    banner.textContent = "Unable to check the server data loading status.";
  }
}
monitorDatasetLoading();

if (form) {
  form.addEventListener("submit", (event) => {
    const datasetSource = document.getElementById("dataset_source");
    if (datasetSource && !datasetSource.value) {
      event.preventDefault();
      datasetSource.focus();
      datasetSource.setCustomValidity("Select a validated server dataset source.");
      datasetSource.reportValidity();
    }
  });
}
