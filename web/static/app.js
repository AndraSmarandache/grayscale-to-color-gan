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
let lastData    = null;  // last colorize response, used for lightbox

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

  // populate U-Net diagram thumbnails and wire up interactions
  document.getElementById("unet-thumb-input").src  = b64src(data.input_L);
  document.getElementById("unet-thumb-output").src = b64src(data.colorized_rgb);

  setupUnet(data.feature_maps);

  results.classList.remove("hidden");
  results.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ── U-Net diagram ──────────────────────────────────────────────

// what each arrow/connection type explains
const OP_INFO = {
  downsample: {
    title: "Downsampling — Strided Convolution",
    visual: [
      "Input:  [B, C,  H,  W]",
      "Conv2d: kernel=3, stride=2",
      "Output: [B, 2C, H/2, W/2]",
      "",
      "  ┌───┬───┬───┬───┐",
      "  │ . │ . │ . │ . │  stride 2 →",
      "  ├───┼───┼───┼───┤  samples every other position",
      "  │ . │ ✓ │ . │ ✓ │  spatial size halved",
      "  └───┴───┴───┴───┘  channel depth doubled",
    ].join("\n"),
    desc: "Each encoder step applies a strided convolution that halves the spatial resolution (H×W → H/2×W/2) while increasing the number of channels. This forces the network to compress spatial information into richer feature representations — trading resolution for semantics."
  },
  upsample: {
    title: "Upsampling — PixelShuffle (Sub-pixel Conv)",
    visual: [
      "Input:  [B, C*r², H,   W  ]   r=2",
      "Output: [B, C,    H*r, W*r]",
      "",
      "  Low-res, many channels → High-res, fewer channels",
      "",
      "  ┌──┬──┐          ┌─┬─┬─┬─┐",
      "  │c0│c1│  r²=4    │ │ │ │ │",
      "  │c2│c3│ ───────► │ │ │ │ │",
      "  └──┴──┘ shuffle  └─┴─┴─┴─┘",
      "  2×2 px, 4 ch       4×4 px, 1 ch",
    ].join("\n"),
    desc: "Sub-pixel convolution (PixelShuffle) rearranges elements from a low-resolution feature map with many channels into a high-resolution map with fewer channels. Compared to transposed convolutions it avoids checkerboard artifacts because it operates in frequency space."
  },
  skip: {
    title: "Skip Connection — Concatenation",
    visual: [
      "  Encoder output  ──────────────────────────┐",
      "  at this depth                              │ cat()",
      "                                             ▼",
      "  Decoder (upsampled) ──► [ enc_feat | dec_feat ]",
      "                           └──────────────────┘",
      "                              combined tensor",
      "                           channels are doubled",
    ].join("\n"),
    desc: "Skip connections copy encoder feature maps directly to the corresponding decoder level and concatenate them channel-wise. This gives the decoder direct access to fine-grained spatial detail (edges, textures) that would otherwise be lost through the bottleneck — critical for colorization where local texture must be preserved."
  },
  bottleneck_down: {
    title: "Encoder → Bottleneck",
    visual: [
      "  Encoder 4:   [B, 512, 16, 16]",
      "  Conv 3×3 ×2",
      "  Bottleneck:  [B, 512,  8,  8]",
      "",
      "  Most compressed representation.",
      "  The network must encode the entire",
      "  scene context into an 8×8 grid.",
    ].join("\n"),
    desc: "The final downsampling step produces the bottleneck — the most compressed representation in the network. At 8×8 spatial resolution, each 'pixel' of the bottleneck covers a 32×32 receptive field in the original input. This is where the network decides what color to use based on high-level scene understanding."
  },
  bottleneck_up: {
    title: "Bottleneck → Decoder",
    visual: [
      "  Bottleneck: [B, 512, 8,  8 ]",
      "  PixelShuffle r=2",
      "  Decoder 4:  [B, 512, 16, 16]",
      "  + skip from Encoder 4",
      "  → concat:   [B,1024, 16, 16]",
      "  Conv 3×3",
      "  → output:   [B, 512, 16, 16]",
    ].join("\n"),
    desc: "The bottleneck output is upsampled back to 16×16 and immediately merged with the skip connection from Encoder 4. This is the first step of color prediction — the high-level semantic context from the bottleneck is combined with the spatial detail preserved in the skip connection."
  },
  output_conv: {
    title: "Output Projection — Conv 1×1 + Tanh",
    visual: [
      "  Decoder 1: [B, 64, 128, 128]",
      "  Conv 1×1",
      "  Output:    [B,  2, 256, 256]  ← final upsample",
      "  Tanh activation",
      "  Values in [-1, 1]",
      "",
      "  Channel 0 → a (green↔red)",
      "  Channel 1 → b (blue↔yellow)",
    ].join("\n"),
    desc: "A 1×1 convolution collapses the 64-channel decoder output to exactly 2 channels — the predicted a and b color channels. Tanh squashes the output to [-1, 1], matching the normalization used during training. These two channels are then rescaled by 110 and merged with the original L channel to produce the final RGB image."
  },
};

let activeBlockKey  = null;
let fmapData        = {};

function setupUnet(featureMaps) {
  fmapData = featureMaps || {};

  // wire block clicks
  document.querySelectorAll(".ublock[data-key]").forEach(el => {
    el.addEventListener("click", () => openFmapPanel(el.dataset.key, el));
  });

  // draw SVG connections after layout settles
  requestAnimationFrame(() => requestAnimationFrame(drawUnetSvg));
}

function drawUnetSvg() {
  const svg  = document.getElementById("unet-svg");
  const diag = document.getElementById("unet-diagram");
  svg.innerHTML = "";
  // allow pointer events through SVG — individual decorative lines are set to none below
  svg.style.pointerEvents = "none";

  const dr = diag.getBoundingClientRect();

  function pos(id) {
    const r = document.getElementById(id).getBoundingClientRect();
    return {
      x:  r.left - dr.left,
      y:  r.top  - dr.top,
      cx: r.left - dr.left + r.width  / 2,
      cy: r.top  - dr.top  + r.height / 2,
      w:  r.width,
      h:  r.height,
      r:  r.right - dr.left,
      b:  r.bottom - dr.top,
    };
  }

  // pairs: enc → enc (downsample), enc4 → bottleneck, bottleneck → dec4, dec → dec (upsample), skip connections
  const encIds = ["ub-enc1","ub-enc2","ub-enc3","ub-enc4"];
  const decIds = ["ub-dec1","ub-dec2","ub-dec3","ub-dec4"];  // dec1 = shallowest

  // encoder vertical arrows (downsample)
  for (let i = 0; i < encIds.length - 1; i++) {
    const a = pos(encIds[i]);
    const b = pos(encIds[i + 1]);
    drawArrow(svg, a.cx, a.b, b.cx, b.y, "#4f8ef7", "downsample");
  }

  // enc4 → bottleneck
  const enc4 = pos("ub-enc4");
  const bot  = pos("ub-bottleneck");
  drawArrow(svg, enc4.cx, enc4.b, bot.cx, bot.y, "#22d3ee", "bottleneck_down");

  // bottleneck → dec4
  const dec4 = pos("ub-dec4");
  drawArrow(svg, bot.cx, bot.y, dec4.cx, dec4.b, "#818cf8", "bottleneck_up");

  // decoder vertical arrows (upsample) — dec4 is deepest, dec1 shallowest
  for (let i = decIds.length - 1; i > 0; i--) {
    const a = pos(decIds[i]);
    const b = pos(decIds[i - 1]);
    drawArrow(svg, a.cx, a.y, b.cx, b.b, "#818cf8", "upsample");
  }

  // skip connections — horizontal dashes from enc to matching dec
  // enc1↔dec1, enc2↔dec2, enc3↔dec3, enc4↔dec4
  const skipPairs = [
    ["ub-enc1","ub-dec1"],
    ["ub-enc2","ub-dec2"],
    ["ub-enc3","ub-dec3"],
    ["ub-enc4","ub-dec4"],
  ];
  skipPairs.forEach(([eid, did]) => {
    const e = pos(eid);
    const d = pos(did);
    drawSkip(svg, e.r, e.cy, d.x, d.cy);
  });

  // output arrow: dec1 → output block
  const dec1   = pos("ub-dec1");
  const output = pos("ub-output");
  drawArrow(svg, dec1.cx, dec1.y, output.cx, output.b, "#22d3ee", "output_conv");

  // input → enc1
  const input = pos("ub-input");
  const enc1  = pos("ub-enc1");
  drawArrow(svg, input.cx, input.b, enc1.cx, enc1.y, "#4f8ef7", "downsample");
}

function drawArrow(svg, x1, y1, x2, y2, color, opKey) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
  g.style.pointerEvents = "all";  // override SVG parent none

  // visible arrow line (decorative, no pointer events)
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x2); line.setAttribute("y2", y2);
  line.setAttribute("stroke", color);
  line.setAttribute("stroke-width", "2.5");
  line.setAttribute("stroke-opacity", "0.6");
  line.setAttribute("marker-end", `url(#arr-${color.replace("#","")})`);
  line.style.pointerEvents = "none";

  // wide invisible hit area on top
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

  // small label near midpoint
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  // offset label to the side so it doesn't overlap the arrow
  const dx = y2 - y1, dy = -(x2 - x1);
  const len = Math.sqrt(dx*dx + dy*dy) || 1;
  const lx = mx + (dx/len) * 10;
  const ly = my + (dy/len) * 10;
  const label = OP_INFO[opKey] ? OP_INFO[opKey].title.split("—")[0].trim() : "";
  if (label) {
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", lx);
    text.setAttribute("y", ly);
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

// redraw arrows on resize
window.addEventListener("resize", () => {
  if (!results.classList.contains("hidden")) {
    requestAnimationFrame(drawUnetSvg);
  }
});


// ── info panel ─────────────────────────────────────────────────

function showInfoPanel(opKey) {
  const info  = OP_INFO[opKey];
  if (!info) return;

  closeFmapPanel();

  document.getElementById("info-title").textContent   = info.title;
  document.getElementById("info-visual").textContent  = info.visual;
  document.getElementById("info-desc").textContent    = info.desc;

  const panel = document.getElementById("info-panel");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("info-close").addEventListener("click", () => {
  document.getElementById("info-panel").classList.add("hidden");
});


// ── feature map panel ──────────────────────────────────────────

let lbImages = [];
let lbIndex  = 0;

function openFmapPanel(key, blockEl) {
  const fm = fmapData[key];
  if (!fm) return;

  closeInfoPanel();

  document.querySelectorAll(".ublock").forEach(b => b.classList.remove("active"));
  if (blockEl) blockEl.classList.add("active");
  activeBlockKey = key;

  document.getElementById("fmap-title").textContent = fm.label;
  document.getElementById("fmap-desc").textContent  = fm.desc;
  document.getElementById("fmap-stats").textContent =
    `${fm.channels.length} of ${fm.total_ch} channels shown · ${fm.spatial}`;

  const grid = document.getElementById("fmap-panel-grid");
  grid.innerHTML = "";
  lbImages = fm.channels.map(b64src);

  // build deck of cards: card 0 = back (first DOM, rendered under), card n-1 = front (last DOM, on top)
  const scene = document.createElement("div");
  scene.className = "deck-scene";

  const deck = document.createElement("div");
  deck.className = "deck";

  const n = fm.channels.length;
  const maxI = Math.max(n - 1, 1);

  fm.channels.forEach((ch, i) => {
    const card = document.createElement("div");
    card.className = "deck-card";
    // normalize to 0-15 range so CSS transform values stay consistent
    card.style.setProperty("--i", (i / maxI) * 15);

    const img = document.createElement("img");
    img.src = b64src(ch);
    img.alt = `ch ${i}`;
    card.appendChild(img);
    card.addEventListener("click", (e) => { e.stopPropagation(); openLightbox(i); });
    deck.appendChild(card);
  });

  const hint = document.createElement("p");
  hint.className = "deck-hint";
  hint.textContent = "Hover to fan out, click any filter to expand";

  scene.appendChild(deck);
  scene.appendChild(hint);
  grid.appendChild(scene);

  const panel = document.getElementById("fmap-panel");
  panel.classList.remove("hidden");
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

document.getElementById("fmap-back").addEventListener("click", closeFmapPanel);

function closeFmapPanel() {
  document.getElementById("fmap-panel").classList.add("hidden");
  document.querySelectorAll(".ublock").forEach(b => b.classList.remove("active"));
  activeBlockKey = null;
}

function closeInfoPanel() {
  document.getElementById("info-panel").classList.add("hidden");
}


// ── lightbox ───────────────────────────────────────────────────

const lightbox   = document.getElementById("lightbox");
const lbImg      = document.getElementById("lightbox-img");
const lbCounter  = document.getElementById("lb-counter");

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
  if (lightbox.classList.contains("hidden")) return;
  if (e.key === "ArrowLeft")  { lbIndex = (lbIndex - 1 + lbImages.length) % lbImages.length; updateLightbox(); }
  if (e.key === "ArrowRight") { lbIndex = (lbIndex + 1) % lbImages.length; updateLightbox(); }
  if (e.key === "Escape")     { lightbox.classList.add("hidden"); }
});
