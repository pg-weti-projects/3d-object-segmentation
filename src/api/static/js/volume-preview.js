const previewSlices = window.volumePreviewData;
const canvas = document.getElementById("preview");
const slider = document.getElementById("slice-slider");
const sliceLabel = document.getElementById("slice-label");

function renderSlice(index) {
  const slice = previewSlices[index];
  const height = slice.length;
  const width = slice[0].length;
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);

  slice.forEach((row, y) => row.forEach((value, x) => {
    const color = Math.round(value * 255);
    const offset = (y * width + x) * 4;
    image.data[offset] = color;
    image.data[offset + 1] = color;
    image.data[offset + 2] = color;
    image.data[offset + 3] = 255;
  }));

  context.putImageData(image, 0, 0);
  sliceLabel.textContent = `(Z=${index})`;
}

if (previewSlices && previewSlices.length && canvas && slider && sliceLabel) {
  renderSlice(Number(slider.value));
  slider.addEventListener("input", () => renderSlice(Number(slider.value)));
}
