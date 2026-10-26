// Three-plane preview of a volume.
//
// It takes its data from whichever of the two sources is present:
//   * the preprocessing page puts it on the container as data-volume-* attributes
//     and may give a URL to fetch one slice at a time instead of inlining them;
//   * the segmentation page puts it in window.volumePreview* globals, together
//     with a mask to overlay and a seed point to show and to pick by clicking.
(() => {
const container = document.querySelector("[data-volume-slices]");
const attributes = container ? container.dataset : {};
const parse = (value, fallback) => {
  if (value === undefined || value === null) return fallback;
  try {
    const parsed = JSON.parse(value);
    return parsed === null ? fallback : parsed;
  } catch (error) {
    return fallback;
  }
};

const inlineSlices = window.volumePreviewData || parse(attributes.volumeSlices, null);
const spacing = window.volumePreviewSpacing || parse(attributes.volumeSpacing, [1, 1, 1]);
const previewUrl = parse(attributes.volumePreviewUrl, null);
const attributeShape = parse(attributes.volumeShape, [0, 0, 0]);
const mask = window.volumePreviewMask || null;
let seed = window.volumePreviewSeed || null;
const pickInput = window.volumePreviewPickSeed
  ? document.getElementById(window.volumePreviewPickSeed) : null;
const MASK_OPACITY = 0.45;
const MIN_DISPLAY_SIZE = 512;

if (!inlineSlices && !previewUrl) {
  return;
}

const depth = inlineSlices ? inlineSlices.length : attributeShape[0];
const height = inlineSlices ? inlineSlices[0].length : attributeShape[1];
const width = inlineSlices ? inlineSlices[0][0].length : attributeShape[2];
// On the segmentation page a large volume is previewed every step-th voxel, and
// the seed is given in full-volume voxels.
const step = window.volumePreviewStep || [1, 1, 1];
const fullShape = window.volumePreviewShape || [depth, height, width];
const toPreview = (point) => point.map((v, i) => Math.floor(v / step[i]));
const toFull = (point) => point.map((v, i) => Math.min(v * step[i], fullShape[i] - 1));
const previewSeed = () => seed && toPreview(seed);
const [spacingZ, spacingY, spacingX] = spacing;

const planes = {
  axial: {
    axis: "Z", axisIndex: 0, count: depth, columns: width, rows: height,
    columnSpacing: spacingX, rowSpacing: spacingY,
    value: (index, row, column) => inlineSlices[index][row][column],
    masked: (index, row, column) => mask[index][row][column],
    seed: () => { const s = previewSeed(); return s && [s[0], s[1], s[2]]; },
    toSeed: (index, row, column) => [index, row, column],
  },
  coronal: {
    axis: "Y", axisIndex: 1, count: height, columns: width, rows: depth,
    columnSpacing: spacingX, rowSpacing: spacingZ,
    value: (index, row, column) => inlineSlices[row][index][column],
    masked: (index, row, column) => mask[row][index][column],
    seed: () => { const s = previewSeed(); return s && [s[1], s[0], s[2]]; },
    toSeed: (index, row, column) => [row, index, column],
  },
  sagittal: {
    axis: "X", axisIndex: 2, count: width, columns: height, rows: depth,
    columnSpacing: spacingY, rowSpacing: spacingZ,
    value: (index, row, column) => inlineSlices[row][column][index],
    masked: (index, row, column) => mask[row][column][index],
    seed: () => { const s = previewSeed(); return s && [s[2], s[0], s[1]]; },
    toSeed: (index, row, column) => [row, column, index],
  },
};

Object.entries(planes).forEach(([name, plane]) => {
  const canvas = document.getElementById(`${name}-canvas`);
  const slider = document.getElementById(`${name}-slider`);
  const label = document.getElementById(`${name}-label`);
  if (!canvas || !slider || !label) {
    return;
  }

  const source = document.createElement("canvas");
  source.width = plane.columns;
  source.height = plane.rows;
  const sourceContext = source.getContext("2d");
  const image = sourceContext.createImageData(plane.columns, plane.rows);

  const physicalWidth = plane.columns * plane.columnSpacing;
  const physicalHeight = plane.rows * plane.rowSpacing;
  const scale = Math.max(plane.columns, plane.rows, MIN_DISPLAY_SIZE)
    / Math.max(physicalWidth, physicalHeight);
  canvas.width = Math.max(1, Math.round(physicalWidth * scale));
  canvas.height = Math.max(1, Math.round(physicalHeight * scale));
  const context = canvas.getContext("2d");
  let loadedIndex = null;
  let requestController = null;

  function caption(index) {
    const factor = step[plane.axisIndex];
    return `(${plane.axis}=${index * factor} / ${(plane.count - 1) * factor})`;
  }

  function renderFetched(index) {
    if (loadedIndex === index) {
      return;
    }
    if (requestController) {
      requestController.abort();
    }
    requestController = new AbortController();
    fetch(`${previewUrl}?plane=${name}&index=${index}`, {signal: requestController.signal})
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load preview slice.");
        }
        return response.blob();
      })
      .then((blob) => createImageBitmap(blob))
      .then((bitmap) => {
        loadedIndex = index;
        context.clearRect(0, 0, canvas.width, canvas.height);
        context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        bitmap.close();
        label.textContent = caption(index);
      })
      .catch((error) => {
        if (error.name === "AbortError") return;
        label.textContent = "Preview unavailable";
      });
  }

  function render(index) {
    if (!inlineSlices) {
      renderFetched(index);
      return;
    }
    for (let row = 0; row < plane.rows; row += 1) {
      for (let column = 0; column < plane.columns; column += 1) {
        const value = Math.round(plane.value(index, row, column) * 255);
        const pixel = (row * plane.columns + column) * 4;
        if (mask && plane.masked(index, row, column)) {
          image.data[pixel] = Math.round(value * (1 - MASK_OPACITY) + 255 * MASK_OPACITY);
          image.data[pixel + 1] = Math.round(value * (1 - MASK_OPACITY));
          image.data[pixel + 2] = Math.round(value * (1 - MASK_OPACITY));
        } else {
          image.data[pixel] = value;
          image.data[pixel + 1] = value;
          image.data[pixel + 2] = value;
        }
        image.data[pixel + 3] = 255;
      }
    }
    sourceContext.putImageData(image, 0, 0);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    drawSeed(index);
    label.textContent = caption(index);
  }

  function drawSeed(index) {
    const point = plane.seed();
    if (!point || point[0] !== index) {
      return;
    }
    const x = (point[2] + 0.5) * canvas.width / plane.columns;
    const y = (point[1] + 0.5) * canvas.height / plane.rows;
    const size = Math.max(6, canvas.width / 40);
    context.strokeStyle = "#ffd600";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x - size, y);
    context.lineTo(x + size, y);
    context.moveTo(x, y - size);
    context.lineTo(x, y + size);
    context.stroke();
  }

  function show(index) {
    const clamped = Math.min(Math.max(index, 0), plane.count - 1);
    slider.value = clamped;
    render(clamped);
  }

  slider.addEventListener("input", () => render(Number(slider.value)));
  plane.show = show;

  document.querySelectorAll(`.slice-step[data-plane="${name}"]`).forEach((button) => {
    button.addEventListener("click", () => {
      show(Number(slider.value) + Number(button.dataset.step));
    });
  });

  if (pickInput && inlineSlices) {
    canvas.style.cursor = "crosshair";
    canvas.addEventListener("click", (event) => {
      const box = canvas.getBoundingClientRect();
      const column = Math.floor((event.clientX - box.left) / box.width * plane.columns);
      const row = Math.floor((event.clientY - box.top) / box.height * plane.rows);
      if (column < 0 || row < 0 || column >= plane.columns || row >= plane.rows) {
        return;
      }
      setSeed(toFull(plane.toSeed(Number(slider.value), row, column)), true);
    });
  }

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    if (event.deltaY !== 0) {
      show(Number(slider.value) + Math.sign(event.deltaY));
    }
  }, { passive: false });

  show(Number(slider.value));
});

function setSeed(point, updateInput) {
  seed = point;
  if (updateInput && pickInput) {
    pickInput.value = point.join(",");
  }
  Object.values(planes).forEach((plane) => {
    if (plane.show) {
      plane.show(toPreview(point)[plane.axisIndex]);
    }
  });
}

if (pickInput) {
  pickInput.addEventListener("change", () => {
    const parts = pickInput.value.split(/[,;\s]+/).filter(Boolean).map(Number);
    if (parts.length === 3 && parts.every((v, i) => Number.isInteger(v) && v >= 0 && v < fullShape[i])) {
      setSeed(parts, false);
    }
  });
}
})();
