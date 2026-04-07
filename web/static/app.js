const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const previewImg  = document.getElementById("preview-img");
const colorizeBtn = document.getElementById("colorize-btn");
const resetBtn    = document.getElementById("reset-btn");
const spinner     = document.getElementById("spinner");
const results     = document.getElementById("results");
const uncSection  = document.getElementById("uncertainty-section");

let currentFile = null;

// drag and drop
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) showPreview(file);
});
dropZone.addEventListener("click", e => {
  // don't trigger if the user clicked the file label (it opens the dialog itself)
  if (e.target.closest(".file-label")) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) showPreview(fileInput.files[0]); });

function showPreview(file) {
  currentFile = file;
  previewImg.src = URL.createObjectURL(file);
  dropZone.classList.add("hidden");
  previewArea.classList.remove("hidden");
  results.classList.add("hidden");
}

resetBtn.addEventListener("click", () => {
  currentFile = null;
  fileInput.value = "";
  dropZone.classList.remove("hidden");
  previewArea.classList.add("hidden");
  results.classList.add("hidden");
});

colorizeBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  spinner.classList.remove("hidden");
  colorizeBtn.disabled = true;
  results.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const resp = await fetch("/api/colorize", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      alert("Error: " + (err.detail || resp.statusText));
      return;
    }
    const data = await resp.json();
    renderResults(data);
  } catch (e) {
    alert("Request failed: " + e.message);
  } finally {
    spinner.classList.add("hidden");
    colorizeBtn.disabled = false;
  }
});

function b64src(b64) {
  return "data:image/png;base64," + b64;
}

function renderResults(data) {
  // LAB decomposition panel
  document.getElementById("img-L").src         = b64src(data.input_L);
  document.getElementById("img-a").src         = b64src(data.ab_a);
  document.getElementById("img-b").src         = b64src(data.ab_b);
  document.getElementById("img-colorized").src = b64src(data.colorized_rgb);

  // final comparison
  document.getElementById("img-original").src  = b64src(data.original_rgb);
  document.getElementById("img-result").src    = b64src(data.colorized_rgb);

  // uncertainty — only shown if the uncertainty model is loaded
  if (data.has_uncertainty && data.uncertainty_heatmap) {
    document.getElementById("img-uncertainty-heat").src    = b64src(data.uncertainty_heatmap);
    document.getElementById("img-uncertainty-overlay").src = b64src(data.uncertainty_overlay);
    uncSection.classList.remove("hidden");
  } else {
    uncSection.classList.add("hidden");
  }

  // feature maps per encoder group
  const container = document.getElementById("fmap-container");
  container.innerHTML = "";

  for (const key of ["layer1", "layer2", "layer3", "layer4"]) {
    const group = data.feature_maps[key];
    if (!group) continue;

    const groupEl = document.createElement("div");
    groupEl.className = "fmap-group";
    groupEl.innerHTML = `
      <div class="fmap-group-header">
        <div class="fmap-group-title">${group.label}</div>
        <div class="fmap-group-desc">${group.desc}</div>
      </div>
    `;

    const grid = document.createElement("div");
    grid.className = "fmap-grid";

    for (const channelB64 of group.channels) {
      const thumb = document.createElement("div");
      thumb.className = "fmap-thumb";
      const img = document.createElement("img");
      img.src = b64src(channelB64);
      img.alt = "";
      thumb.appendChild(img);
      // click to enlarge
      thumb.addEventListener("click", () => openLightbox(b64src(channelB64)));
      grid.appendChild(thumb);
    }

    groupEl.appendChild(grid);
    container.appendChild(groupEl);
  }

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openLightbox(src) {
  const lb = document.createElement("div");
  lb.className = "lightbox";
  const img = document.createElement("img");
  img.src = src;
  lb.appendChild(img);
  lb.addEventListener("click", () => lb.remove());
  document.body.appendChild(lb);
}
