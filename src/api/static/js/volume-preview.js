const previewData = window.volumePreviewData;
const canvas = document.getElementById("preview");

if (previewData && canvas) {
  const height = previewData.length;
  const width = previewData[0].length;
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  const image = context.createImageData(width, height);

  previewData.forEach((row, y) => row.forEach((value, x) => {
    const color = Math.round(value * 255);
    const offset = (y * width + x) * 4;
    image.data[offset] = color;
    image.data[offset + 1] = color;
    image.data[offset + 2] = color;
    image.data[offset + 3] = 255;
  }));

  context.putImageData(image, 0, 0);
}
