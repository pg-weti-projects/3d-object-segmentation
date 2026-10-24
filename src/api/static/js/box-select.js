// Box around the swim bladder on the axial image of the segmentation page.
//
// Drag on the axial image (on the slice where the bladder is largest) to draw a
// rectangle. It is written to the hidden fields box_x_min, box_x_max, box_y_min,
// box_y_max of the form (voxel indices of the preprocessed volume, max exclusive),
// and the seed point is set to the centre of the box on the current slice. A click
// without dragging still sets the seed point (volume-preview.js).
//
// Uses window.volumePreviewData / volumePreviewStep / volumePreviewShape.
(() => {
const slices = window.volumePreviewData;
const canvas = document.getElementById("axial-canvas");
const slider = document.getElementById("axial-slider");
const overlay = document.getElementById("box-overlay");
const status = document.getElementById("box-status");
const clearButton = document.getElementById("box-clear");
const seedInput = document.getElementById("seed_zyx");
const field = (name) => document.querySelector(`input[name="${name}"]`);
const fields = { x0: field("box_x_min"), x1: field("box_x_max"), y0: field("box_y_min"), y1: field("box_y_max") };
if (!slices || !slices.length || !canvas || !slider || !overlay || !status || !Object.values(fields).every(Boolean)) {
  return;
}

const rows = slices[0].length;
const columns = slices[0][0].length;
const step = window.volumePreviewStep || [1, 1, 1];
const fullShape = window.volumePreviewShape || [slices.length, rows, columns];
const DRAG_PIXELS = 4;
const defaultStatus = status.textContent;

let box = readFields(); // full-volume voxel indices: { x0, x1, y0, y1 }
let drag = null;
let dragged = false;

function readFields() {
  const values = Object.values(fields).map((input) => input.value.trim());
  if (values.some((value) => value === "" || !Number.isInteger(Number(value)))) {
    return null;
  }
  const [x0, x1, y0, y1] = Object.keys(fields).map((key) => Number(fields[key].value));
  return x1 > x0 && y1 > y0 ? { x0, x1, y0, y1 } : null;
}

function clamp(value, low, high) {
  return Math.min(Math.max(value, low), high);
}

function previewPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    column: clamp(((event.clientX - rect.left) / rect.width) * columns, 0, columns),
    row: clamp(((event.clientY - rect.top) / rect.height) * rows, 0, rows),
    pixelX: event.clientX,
    pixelY: event.clientY,
  };
}

function boxFromPreview(a, b) {
  const x0 = Math.floor(Math.min(a.column, b.column));
  const y0 = Math.floor(Math.min(a.row, b.row));
  const x1 = Math.max(Math.ceil(Math.max(a.column, b.column)), x0 + 1);
  const y1 = Math.max(Math.ceil(Math.max(a.row, b.row)), y0 + 1);
  return {
    x0: Math.min(x0 * step[2], fullShape[2] - 1),
    x1: Math.min(x1 * step[2], fullShape[2]),
    y0: Math.min(y0 * step[1], fullShape[1] - 1),
    y1: Math.min(y1 * step[1], fullShape[1]),
  };
}

function placeOverlay() {
  if (!box) {
    overlay.hidden = true;
    return;
  }
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  overlay.style.left = `${canvas.offsetLeft + (box.x0 / fullShape[2]) * width}px`;
  overlay.style.top = `${canvas.offsetTop + (box.y0 / fullShape[1]) * height}px`;
  overlay.style.width = `${((box.x1 - box.x0) / fullShape[2]) * width}px`;
  overlay.style.height = `${((box.y1 - box.y0) / fullShape[1]) * height}px`;
  overlay.hidden = false;
}

function update(setSeed) {
  placeOverlay();
  const keys = Object.keys(fields);
  keys.forEach((key) => (fields[key].value = box ? String(box[key]) : ""));
  clearButton.hidden = !box;
  status.textContent = box
    ? `x ${box.x0}–${box.x1 - 1}, y ${box.y0}–${box.y1 - 1} (${box.x1 - box.x0} × ${box.y1 - box.y0} voxels, all slices)`
    : defaultStatus;
  if (box && setSeed && seedInput) {
    const z = Math.min(Number(slider.value) * step[0], fullShape[0] - 1);
    seedInput.value = [z, Math.floor((box.y0 + box.y1 - 1) / 2), Math.floor((box.x0 + box.x1 - 1) / 2)].join(",");
    seedInput.dispatchEvent(new Event("change"));
  }
}

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) {
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  drag = previewPoint(event);
  dragged = false;
});

canvas.addEventListener("pointermove", (event) => {
  if (!drag) {
    return;
  }
  const point = previewPoint(event);
  if (!dragged && Math.hypot(point.pixelX - drag.pixelX, point.pixelY - drag.pixelY) < DRAG_PIXELS) {
    return;
  }
  dragged = true;
  box = boxFromPreview(drag, point);
  update(false);
});

["pointerup", "pointercancel"].forEach((name) =>
  canvas.addEventListener(name, () => {
    if (drag && dragged) {
      update(true);
    }
    drag = null;
  }),
);

// A drag ends with a click on the canvas; do not let it move the seed point.
window.addEventListener(
  "click",
  (event) => {
    if (dragged && event.target === canvas) {
      event.stopPropagation();
    }
    dragged = false;
  },
  true,
);

clearButton.addEventListener("click", () => {
  box = null;
  update(false);
});

new ResizeObserver(placeOverlay).observe(canvas);
window.addEventListener("resize", placeOverlay);
update(false);
})();
