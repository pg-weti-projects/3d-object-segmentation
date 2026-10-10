const fileInput = document.getElementById("volume");
const form = document.getElementById("preprocess-form");

if (fileInput) {
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      const help = fileInput.nextElementSibling;
      if (help) {
        help.textContent = `${fileInput.files.length} file(s) selected.`;
      }
    }
  });

  fileInput.addEventListener("dragover", (event) => {
    event.preventDefault();
    fileInput.classList.add("border-primary");
  });

  fileInput.addEventListener("dragleave", () => {
    fileInput.classList.remove("border-primary");
  });

  fileInput.addEventListener("drop", (event) => {
    event.preventDefault();
    fileInput.classList.remove("border-primary");
    const items = [...event.dataTransfer.items];
    const entries = items
      .map((item) => item.webkitGetAsEntry && item.webkitGetAsEntry())
      .filter(Boolean);

    if (entries.some((entry) => entry.isDirectory)) {
      readEntries(entries).then((files) => {
        const transfer = new DataTransfer();
        files.forEach((file) => transfer.items.add(file));
        fileInput.files = transfer.files;
        fileInput.dispatchEvent(new Event("change"));
      });
    } else if (event.dataTransfer.files.length) {
      fileInput.files = event.dataTransfer.files;
      fileInput.dispatchEvent(new Event("change"));
    }
  });
}

async function readEntries(entries) {
  const files = [];
  for (const entry of entries) {
    if (entry.isFile) {
      files.push(await readFile(entry));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let children = [];
      let batch;
      do {
        batch = await readDirectoryBatch(reader);
        children = children.concat(batch);
      } while (batch.length);
      files.push(...await readEntries(children));
    }
  }
  return files;
}

function readFile(entry) {
  return new Promise((resolve, reject) => entry.file(resolve, reject));
}

function readDirectoryBatch(reader) {
  return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

if (form) {
  form.addEventListener("submit", (event) => {
    if (!fileInput.files.length) {
      event.preventDefault();
      fileInput.focus();
      fileInput.setCustomValidity("Select an input file or a folder of slices.");
      fileInput.reportValidity();
    } else {
      fileInput.setCustomValidity("");
    }
  });
}
