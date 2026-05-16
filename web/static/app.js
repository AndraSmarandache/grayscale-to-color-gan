const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const previewImg  = document.getElementById("preview-img");
const colorizeBtn = document.getElementById("colorize-btn");
const compareBtn  = document.getElementById("compare-btn");
const resetBtn    = document.getElementById("reset-btn");
const spinner     = document.getElementById("spinner");
const results     = document.getElementById("results");
const uncSection  = document.getElementById("uncertainty-section");

let currentFile = null;
let lastData    = null;

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
  document.getElementById("compare-section").classList.add("hidden");
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
    lastData = await resp.json();
    renderResults(lastData);
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
  document.getElementById("img-L").src         = b64src(data.input_L);
  document.getElementById("img-a").src         = b64src(data.ab_a);
  document.getElementById("img-b").src         = b64src(data.ab_b);
  document.getElementById("img-colorized").src = b64src(data.colorized_rgb);
  document.getElementById("img-original").src  = b64src(data.original_rgb);
  document.getElementById("img-result").src    = b64src(data.colorized_rgb);

  if (data.has_uncertainty && data.uncertainty_heatmap) {
    document.getElementById("img-uncertainty-heat").src    = b64src(data.uncertainty_heatmap);
    document.getElementById("img-uncertainty-overlay").src = b64src(data.uncertainty_overlay);
    uncSection.classList.remove("hidden");
  } else {
    uncSection.classList.add("hidden");
  }

  document.getElementById("unet-thumb-input").src  = b64src(data.input_L);
  document.getElementById("unet-thumb-output").src = b64src(data.colorized_rgb);

  setupUnet(data.feature_maps);

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}


// model selector
const modelSelect = document.getElementById("model-select");
const modelBadge  = document.getElementById("model-badge");
const modelStatus = document.getElementById("model-status");

async function loadModelList() {
  try {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    modelSelect.innerHTML = "";
    data.models.forEach(m => {
      const opt = document.createElement("option");
      opt.value       = JSON.stringify({ path: m.path, uncertainty: m.uncertainty });
      opt.textContent = m.name;
      if (m.path === data.current_path) opt.selected = true;
      modelSelect.appendChild(opt);
    });
    updateBadge(data.uncertainty);
  } catch (e) {
    modelStatus.textContent = "could not load model list";
  }
}

function updateBadge(isUncertainty) {
  if (isUncertainty) {
    modelBadge.classList.remove("hidden");
  } else {
    modelBadge.classList.add("hidden");
  }
}

modelSelect.addEventListener("change", async () => {
  const val = JSON.parse(modelSelect.value);
  modelStatus.textContent = "loading...";
  modelSelect.disabled    = true;
  try {
    const resp = await fetch("/api/load_model", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(val),
    });
    if (!resp.ok) {
      const err = await resp.json();
      modelStatus.textContent = "error: " + (err.detail || "failed");
      return;
    }
    const data = await resp.json();
    updateBadge(data.uncertainty);
    modelStatus.textContent = "loaded";
    setTimeout(() => { modelStatus.textContent = ""; }, 2000);
  } catch (e) {
    modelStatus.textContent = "request failed";
  } finally {
    modelSelect.disabled = false;
  }
});

loadModelList();


// compare all models
compareBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  const section    = document.getElementById("compare-section");
  const grid       = document.getElementById("compare-grid-all");
  const cmpSpinner = document.getElementById("compare-spinner");

  section.classList.remove("hidden");
  grid.innerHTML = "";
  cmpSpinner.classList.remove("hidden");
  compareBtn.disabled = true;
  section.scrollIntoView({ behavior: "smooth", block: "start" });

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const resp = await fetch("/api/compare", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      alert("Compare error: " + (err.detail || resp.statusText));
      return;
    }
    const data = await resp.json();

    const origCard = document.createElement("div");
    origCard.className = "compare-card compare-original";
    origCard.innerHTML =
      `<img src="${b64src(data.original_rgb)}" />` +
      `<span class="label">Original</span>` +
      `<span class="sublabel">color reference</span>`;
    grid.appendChild(origCard);

    data.results.forEach(r => {
      const card = document.createElement("div");
      card.className = "compare-card";
      if (r.error) {
        card.innerHTML =
          `<span class="compare-error">Failed to load</span>` +
          `<span class="label">${r.name}</span>`;
      } else {
        card.innerHTML =
          `<img src="${b64src(r.colorized_rgb)}" />` +
          `<span class="label">${r.name}</span>` +
          (r.uncertainty ? `<span class="sublabel">uncertainty</span>` : "");
      }
      grid.appendChild(card);
    });
  } catch (e) {
    alert("Request failed: " + e.message);
  } finally {
    cmpSpinner.classList.add("hidden");
    compareBtn.disabled = false;
  }
});


// U-Net diagram
const OP_INFO = {
  downsample: {
    title: "Downsampling - Strided Convolution",
    visual: [
      "Input:  [B, C,  H,  W]",
      "Conv2d: kernel=3, stride=2",
      "Output: [B, 2C, H/2, W/2]",
      "",
      "  stride 2 samples every other position",
      "  spatial size halved, channels doubled",
    ].join("\n"),
    desc: "Each encoder step applies a strided convolution that halves the spatial resolution (H x W to H/2 x W/2) while increasing the number of channels. This forces the network to compress spatial information into richer feature representations."
  },
  upsample: {
    title: "Upsampling - PixelShuffle",
    visual: [
      "Input:  [B, C*r^2, H,   W  ]   r=2",
      "Output: [B, C,     H*r, W*r]",
      "",
      "  Low-res, many channels -> High-res, fewer channels",
      "  Sub-pixel conv avoids checkerboard artifacts",
    ].join("\n"),
    desc: "Sub-pixel convolution (PixelShuffle) rearranges elements from a low-resolution feature map with many channels into a high-resolution map with fewer channels. Compared to transposed convolutions it avoids checkerboard artifacts."
  },
  skip: {
    title: "Skip Connection - Concatenation",
    visual: [
      "  Encoder feat  ─────────────────────────┐",
      "                                          cat()",
      "  Decoder feat  ->  [ enc_feat | dec_feat ]",
      "                     channels are doubled",
    ].join("\n"),
    desc: "Skip connections copy encoder feature maps directly to the corresponding decoder level and concatenate them channel-wise. This gives the decoder direct access to fine-grained spatial detail that would otherwise be lost through the bottleneck."
  },
  bottleneck_down: {
    title: "Encoder to Bottleneck",
    visual: [
      "  Encoder 4:   [B, 512, 16, 16]",
      "  Conv 3x3 x2",
      "  Bottleneck:  [B, 512,  8,  8]",
      "",
      "  Most compressed — 8x8 grid covers full scene",
    ].join("\n"),
    desc: "The final downsampling step produces the bottleneck. At 8x8 resolution, each position covers a 32x32 receptive field in the original input. This is where the network decides what color to use based on high-level scene understanding."
  },
  bottleneck_up: {
    title: "Bottleneck to Decoder",
    visual: [
      "  Bottleneck: [B, 512, 8,  8 ]",
      "  PixelShuffle r=2",
      "  Decoder 4:  [B, 512, 16, 16]",
      "  + skip from Encoder 4",
      "  concat:     [B,1024, 16, 16]",
    ].join("\n"),
    desc: "The bottleneck output is upsampled to 16x16 and merged with the skip connection from Encoder 4. High-level semantic context combines with spatial detail preserved in the skip connection."
  },
  output_conv: {
    title: "Output Projection - Conv 1x1 + Tanh",
    visual: [
      "  Decoder 1: [B, 64, 128, 128]",
      "  Conv 1x1",
      "  Output:    [B,  2, 256, 256]",
      "  Tanh -> values in [-1, 1]",
      "",
      "  ch 0 -> a (green/red axis)",
      "  ch 1 -> b (blue/yellow axis)",
    ].join("\n"),
    desc: "A 1x1 convolution collapses 64 channels to exactly 2 — the predicted a and b color channels. Tanh squashes to [-1,1]. These two channels are rescaled by 110 and merged with the original L channel to produce the final RGB image."
  },
};

let activeBlockKey = null;
let fmapData       = {};

function setupUnet(featureMaps) {
  fmapData = featureMaps || {};
  document.querySelectorAll(".ublock[data-key]").forEach(el => {
    el.addEventListener("click", () => openFmapPanel(el.dataset.key, el));
  });
  requestAnimationFrame(() => requestAnimationFrame(drawUnetSvg));
}

function drawUnetSvg() {
  const svg  = document.getElementById("unet-svg");
  const diag = document.getElementById("unet-diagram");
  svg.innerHTML = "";
  svg.style.pointerEvents = "none";

  const dr = diag.getBoundingClientRect();

  function pos(id) {
    const r = document.getElementById(id).getBoundingClientRect();
    return {
      cx: r.left - dr.left + r.width  / 2,
      cy: r.top  - dr.top  + r.height / 2,
      x:  r.left - dr.left,
      r:  r.right  - dr.left,
      y:  r.top  - dr.top,
      b:  r.bottom - dr.top,
    };
  }

  const encIds = ["ub-enc1","ub-enc2","ub-enc3","ub-enc4"];
  const decIds = ["ub-dec1","ub-dec2","ub-dec3","ub-dec4"];

  for (let i = 0; i < encIds.length - 1; i++) {
    const a = pos(encIds[i]), b = pos(encIds[i + 1]);
    drawArrow(svg, a.cx, a.b, b.cx, b.y, "#4f8ef7", "downsample");
  }

  const enc4 = pos("ub-enc4"), bot = pos("ub-bottleneck");
  drawArrow(svg, enc4.cx, enc4.b, bot.cx, bot.y, "#22d3ee", "bottleneck_down");

  const dec4 = pos("ub-dec4");
  drawArrow(svg, bot.cx, bot.y, dec4.cx, dec4.b, "#818cf8", "bottleneck_up");

  for (let i = decIds.length - 1; i > 0; i--) {
    const a = pos(decIds[i]), b = pos(decIds[i - 1]);
    drawArrow(svg, a.cx, a.y, b.cx, b.b, "#818cf8", "upsample");
  }

  [["ub-enc1","ub-dec1"],["ub-enc2","ub-dec2"],["ub-enc3","ub-dec3"],["ub-enc4","ub-dec4"]]
    .forEach(([eid, did]) => {
      const e = pos(eid), d = pos(did);
      drawSkip(svg, e.r, e.cy, d.x, d.cy);
    });

  const dec1 = pos("ub-dec1"), output = pos("ub-output");
  drawArrow(svg, dec1.cx, dec1.y, output.cx, output.b, "#22d3ee", "output_conv");

  const input = pos("ub-input"), enc1 = pos("ub-enc1");
  drawArrow(svg, input.cx, input.b, enc1.cx, enc1.y, "#4f8ef7", "downsample");
}

function drawArrow(svg, x1, y1, x2, y2, color, opKey) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.style.pointerEvents = "all";

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x2); line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", "2.5");
  line.setAttribute("stroke-opacity", "0.6");
  line.setAttribute("marker-end", `url(#arr-${color.replace("#","")})`);
  line.style.pointerEvents = "none";

  const hit = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hit.setAttribute("x1", x1); hit.setAttribute("y1", y1);
  hit.setAttribute("x2", x2); hit.setAttribute("y2", y2);
  hit.setAttribute("stroke", "transparent");
  hit.setAttribute("stroke-width", "28");
  hit.style.cursor = "pointer";
  hit.style.pointerEvents = "all";
  hit.addEventListener("click", (e) => { e.stopPropagation(); showInfoPanel(opKey); });
  hit.addEventListener("mouseenter", () => {
    line.setAttribute("stroke-opacity", "1");
    line.setAttribute("stroke-width", "3.5");
  });
  hit.addEventListener("mouseleave", () => {
    line.setAttribute("stroke-opacity", "0.6");
    line.setAttribute("stroke-width", "2.5");
  });

  const mx  = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const dx  = y2 - y1,        dy = -(x2 - x1);
  const len = Math.sqrt(dx*dx + dy*dy) || 1;
  const label = OP_INFO[opKey] ? OP_INFO[opKey].title.split("-")[0].trim() : "";
  if (label) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", mx + (dx/len) * 10);
    text.setAttribute("y", my + (dy/len) * 10);
    text.setAttribute("fill", color);
    text.setAttribute("font-size", "9");
    text.setAttribute("opacity", "0.55");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dominant-baseline", "middle");
    text.style.pointerEvents = "none";
    text.textContent = label;
    g.appendChild(text);
  }

  ensureMarker(svg, color);
  g.appendChild(line);
  g.appendChild(hit);
  svg.appendChild(g);
}

function drawSkip(svg, x1, y1, x2, y2) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.style.pointerEvents = "all";

  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x2); line.setAttribute("y2", y2);
  line.setAttribute("stroke", "#f0a0c8");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("stroke-opacity", "0.55");
  line.setAttribute("stroke-dasharray", "6 4");
  line.setAttribute("marker-end", "url(#arr-f0a0c8)");
  line.style.pointerEvents = "none";

  const hit = document.createElementNS("http://www.w3.org/2000/svg", "line");
  hit.setAttribute("x1", x1); hit.setAttribute("y1", y1);
  hit.setAttribute("x2", x2); hit.setAttribute("y2", y2);
  hit.setAttribute("stroke", "transparent");
  hit.setAttribute("stroke-width", "24");
  hit.style.cursor = "pointer";
  hit.style.pointerEvents = "all";
  hit.addEventListener("click", (e) => { e.stopPropagation(); showInfoPanel("skip"); });
  hit.addEventListener("mouseenter", () => { line.setAttribute("stroke-opacity","1"); line.setAttribute("stroke-width","3"); });
  hit.addEventListener("mouseleave", () => { line.setAttribute("stroke-opacity","0.55"); line.setAttribute("stroke-width","2"); });

  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", mx); text.setAttribute("y", my - 8);
  text.setAttribute("fill", "#f0a0c8"); text.setAttribute("font-size", "9");
  text.setAttribute("opacity", "0.55"); text.setAttribute("text-anchor", "middle");
  text.style.pointerEvents = "none";
  text.textContent = "skip";
  g.appendChild(text);

  ensureMarker(svg, "#f0a0c8");
  g.appendChild(line);
  g.appendChild(hit);
  svg.appendChild(g);
}

const _markers = new Set();
function ensureMarker(svg, color) {
  const id = "arr-" + color.replace("#", "");
  if (_markers.has(id)) return;
  _markers.add(id);

  let defs = svg.querySelector("defs");
  if (!defs) { defs = document.createElementNS("http://www.w3.org/2000/svg","defs"); svg.prepend(defs); }

  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", id);
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "7");
  marker.setAttribute("markerHeight", "7");
  marker.setAttribute("orient", "auto");

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  path.setAttribute("fill", color);
  marker.appendChild(path);
  defs.appendChild(marker);
}

window.addEventListener("resize", () => {
  if (!results.classList.contains("hidden")) requestAnimationFrame(drawUnetSvg);
});


// info panel
function showInfoPanel(opKey) {
  const info = OP_INFO[opKey];
  if (!info) return;
  closeFmapPanel();
  document.getElementById("info-title").textContent  = info.title;
  document.getElementById("info-visual").textContent = info.visual;
  document.getElementById("info-desc").textContent   = info.desc;
  const panel = document.getElementById("info-panel");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("info-close").addEventListener("click", () => {
  document.getElementById("info-panel").classList.add("hidden");
});


// feature map modal — deck of cards visualization
let lbImages = [];
let lbIndex  = 0;

function setCardTransform(card, normalI, spread) {
  const rotY = -27 + spread * 18;
  const rotX = 11  - spread * 8;
  const x    = (normalI - 15) * (1 - spread) * 0.9 + (normalI - 7.5) * spread * 52;
  const y    = (15 - normalI) * (1 - spread) * 0.9;
  const z    = (normalI - 15) * (1 - spread) * 10;
  card.style.transform =
    `perspective(700px) rotateY(${rotY.toFixed(2)}deg) rotateX(${rotX.toFixed(2)}deg)` +
    ` translateX(${x.toFixed(1)}px) translateY(${y.toFixed(1)}px) translateZ(${z.toFixed(1)}px)`;
}

function openFmapPanel(key, blockEl) {
  const fm = fmapData[key];
  if (!fm) return;

  closeInfoPanel();
  closeFmapPanel();

  document.querySelectorAll(".ublock").forEach(b => b.classList.remove("active"));
  if (blockEl) blockEl.classList.add("active");
  activeBlockKey = key;

  lbImages = fm.channels.map(b64src);

  const n    = fm.channels.length;
  const maxI = Math.max(n - 1, 1);

  const scene = document.createElement("div");
  scene.className = "deck-scene";

  const deck = document.createElement("div");
  deck.className = "deck";

  const cards = [];
  fm.channels.forEach((ch, i) => {
    const card    = document.createElement("div");
    card.className = "deck-card";
    const normalI  = (i / maxI) * 15;
    setCardTransform(card, normalI, 0);

    const img = document.createElement("img");
    img.src   = b64src(ch);
    img.alt   = `ch ${i}`;
    card.appendChild(img);
    card.addEventListener("click", (e) => { e.stopPropagation(); openLightbox(i); });
    deck.appendChild(card);
    cards.push({ el: card, normalI });
  });

  // slide mouse left fans the deck, slide right stacks it
  scene.addEventListener("mousemove", (e) => {
    const r      = scene.getBoundingClientRect();
    const spread = 1 - Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    cards.forEach(c => setCardTransform(c.el, c.normalI, spread));
  });
  scene.addEventListener("mouseleave", () => {
    cards.forEach(c => setCardTransform(c.el, c.normalI, 0));
  });

  const hint = document.createElement("p");
  hint.className = "deck-hint";
  hint.textContent = "Slide mouse left to fan out\nClick any filter to expand";

  scene.appendChild(deck);
  scene.appendChild(hint);

  // modal overlay
  const overlay = document.createElement("div");
  overlay.className = "fmap-modal-overlay";

  const modal = document.createElement("div");
  modal.className = "fmap-modal";

  const header = document.createElement("div");
  header.className = "fmap-modal-header";

  const titles = document.createElement("div");
  titles.className = "fmap-modal-titles";
  titles.innerHTML =
    `<div class="fmap-modal-title">${fm.label}</div>` +
    `<div class="fmap-modal-desc">${fm.desc}</div>`;

  const stats = document.createElement("div");
  stats.className = "fmap-modal-stats";
  stats.textContent = `${n} channels · ${fm.spatial}`;

  const closeBtn = document.createElement("button");
  closeBtn.className = "fmap-modal-close";
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", closeFmapPanel);

  header.appendChild(titles);
  header.appendChild(stats);
  header.appendChild(closeBtn);

  const body = document.createElement("div");
  body.className = "fmap-modal-body";
  body.appendChild(scene);

  modal.appendChild(header);
  modal.appendChild(body);
  overlay.appendChild(modal);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) closeFmapPanel(); });

  document.body.appendChild(overlay);

  // quick preview animation — fans and returns so user knows it's interactive
  let t = 0;
  const preview = setInterval(() => {
    t += 0.05;
    const s = Math.sin(t * Math.PI) * 0.65;
    cards.forEach(c => setCardTransform(c.el, c.normalI, s));
    if (t >= 1) {
      clearInterval(preview);
      cards.forEach(c => setCardTransform(c.el, c.normalI, 0));
    }
  }, 16);
}

document.getElementById("fmap-back").addEventListener("click", closeFmapPanel);

function closeFmapPanel() {
  const overlay = document.querySelector(".fmap-modal-overlay");
  if (overlay) overlay.remove();
  document.querySelectorAll(".ublock").forEach(b => b.classList.remove("active"));
  activeBlockKey = null;
}

function closeInfoPanel() {
  document.getElementById("info-panel").classList.add("hidden");
}


// lightbox
const lightbox  = document.getElementById("lightbox");
const lbImg     = document.getElementById("lightbox-img");
const lbCounter = document.getElementById("lb-counter");

function openLightbox(index) {
  lbIndex = index;
  updateLightbox();
  lightbox.classList.remove("hidden");
}

function updateLightbox() {
  lbImg.src = lbImages[lbIndex];
  lbCounter.textContent = `${lbIndex + 1} / ${lbImages.length}`;
}

document.getElementById("lb-prev").addEventListener("click", e => {
  e.stopPropagation();
  lbIndex = (lbIndex - 1 + lbImages.length) % lbImages.length;
  updateLightbox();
});
document.getElementById("lb-next").addEventListener("click", e => {
  e.stopPropagation();
  lbIndex = (lbIndex + 1) % lbImages.length;
  updateLightbox();
});
lightbox.addEventListener("click", e => {
  if (e.target === lightbox || e.target === lbImg) lightbox.classList.add("hidden");
});
document.addEventListener("keydown", e => {
  if (!lightbox.classList.contains("hidden")) {
    if (e.key === "ArrowLeft")  { lbIndex = (lbIndex - 1 + lbImages.length) % lbImages.length; updateLightbox(); }
    if (e.key === "ArrowRight") { lbIndex = (lbIndex + 1) % lbImages.length; updateLightbox(); }
    if (e.key === "Escape")     { lightbox.classList.add("hidden"); }
  } else if (e.key === "Escape") {
    closeFmapPanel();
  }
});
