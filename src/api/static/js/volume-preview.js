// window.volumePreviewData: nested [Z][Y][X] floats in [0, 1];
// window.volumePreviewSpacing: [z, y, x] voxel size in mm.
(() => {
const previewContainer = document.querySelector("[data-volume-slices]");
const inlineSlices = JSON.parse(previewContainer?.dataset.volumeSlices || "null");
const spacing = JSON.parse(previewContainer?.dataset.volumeSpacing || "[1, 1, 1]");
const previewUrl = JSON.parse(previewContainer?.dataset.volumePreviewUrl || "null");
const MIN_DISPLAY_SIZE = 512;

if (!previewContainer || (!inlineSlices && !previewUrl)) {
  return;
}

const shape = JSON.parse(previewContainer?.dataset.volumeShape || "[0, 0, 0]");
const depth = inlineSlices ? inlineSlices.length : shape[0];
const height = inlineSlices ? inlineSlices[0].length : shape[1];
const width = inlineSlices ? inlineSlices[0][0].length : shape[2];
const [spacingZ, spacingY, spacingX] = spacing;

const planes = {
  axial: {
    axis: "Z", count: depth, columns: width, rows: height,
    columnSpacing: spacingX, rowSpacing: spacingY,
    value: (index, row, column) => inlineSlices[index][row][column],
  },
  coronal: {
    axis: "Y", count: height, columns: width, rows: depth,
    columnSpacing: spacingX, rowSpacing: spacingZ,
    value: (index, row, column) => inlineSlices[row][index][column],
  },
  sagittal: {
    axis: "X", count: width, columns: height, rows: depth,
    columnSpacing: spacingY, rowSpacing: spacingZ,
    value: (index, row, column) => inlineSlices[row][column][index],
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
  let planeData = inlineSlices;
  let loadedIndex = null;
  let requestController = null;

  function render(index) {
    if (!inlineSlices && loadedIndex !== index) {
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
          label.textContent = `(${plane.axis}=${index} / ${plane.count - 1})`;
        })
        .catch((error) => {
          if (error.name === "AbortError") return;
          label.textContent = "Preview unavailable";
        });
      return;
    }
    for (let row = 0; row < plane.rows; row += 1) {
      for (let column = 0; column < plane.columns; column += 1) {
        const value = Math.round((
          inlineSlices ? plane.value(index, row, column) : (planeData[row]?.[column] ?? 0)
        ) * 255);
        const pixel = (row * plane.columns + column) * 4;
        image.data[pixel] = value;
        image.data[pixel + 1] = value;
        image.data[pixel + 2] = value;
        image.data[pixel + 3] = 255;
      }
    }
    sourceContext.putImageData(image, 0, 0);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    label.textContent = `(${plane.axis}=${index} / ${plane.count - 1})`;
  }

  function show(index) {
    const clamped = Math.min(Math.max(index, 0), plane.count - 1);
    slider.value = clamped;
    render(clamped);
  }

  slider.addEventListener("input", () => render(Number(slider.value)));
  document.querySelectorAll(`.slice-step[data-plane="${name}"]`).forEach((button) => {
    button.addEventListener("click", () => {
      show(Number(slider.value) + Number(button.dataset.step));
    });
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    if (event.deltaY !== 0) {
      show(Number(slider.value) + Math.sign(event.deltaY));
    }
  }, { passive: false });

  show(Number(slider.value));
});
})();
