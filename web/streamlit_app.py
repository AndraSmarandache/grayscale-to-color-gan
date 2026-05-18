"""Streamlit demo for the GAN-based image colorization project."""

import base64
import io
import json
import os
import sys

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from skimage import color as skcolor
from skimage.metrics import structural_similarity as ssim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.generator import UncertaintyGenerator, build_res_unet

IMG_SIZE    = 256
MODELS_DIR  = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
MAX_FMAP_CH = 48   # channels shown per layer in the deck viewer
FMAP_SIZE   = 64   # pixel size of each feature-map tile

# model helpers

def is_uncertainty_model(filename: str) -> bool:
    return "uncertainty" in os.path.basename(filename).lower()


@st.cache_data(show_spinner=False)
def get_available_models() -> list[dict]:
    seen, result = set(), []
    for directory in [MODELS_DIR, os.path.join(ROOT, "web")]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith((".pth", ".pt")):
                continue
            fpath = os.path.normpath(os.path.join(directory, fname))
            if fpath in seen:
                continue
            seen.add(fpath)
            result.append({"name": fname, "path": fpath, "uncertainty": is_uncertainty_model(fname)})
    return result


@st.cache_resource(show_spinner=False)
def load_model(path: str, uncertainty: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G  = build_res_unet(n_input=1, n_output=2, size=IMG_SIZE)
    if uncertainty:
        net_G = UncertaintyGenerator(net_G)
    net_G = net_G.to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        for key in ("generator_state_dict", "net_G", "model"):
            if key in state:
                state = state[key]
                break
    net_G.load_state_dict(state, strict=False)
    net_G.eval()
    return net_G


# image helpers

def arr_to_b64(arr: np.ndarray) -> str:
    arr    = np.clip(arr, 0, 1)
    arr_u8 = (arr * 255).astype(np.uint8)
    mode   = "L" if arr.ndim == 2 else "RGB"
    buf    = io.BytesIO()
    Image.fromarray(arr_u8, mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_src(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def _prepare_L(image_bytes: bytes):
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    rgb = np.array(pil).astype(np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb).astype(np.float32)
    L   = lab[:, :, 0]
    return rgb, L, torch.tensor((L / 50.0) - 1.0)[None, None]


def _unc_heatmap(std_norm: np.ndarray) -> np.ndarray:
    return np.stack([std_norm, 1 - np.abs(std_norm * 2 - 1), 1 - std_norm], axis=-1)


# feature map extraction

HOOK_TARGETS = {
    "layers.0.4": "enc1",
    "layers.0.5": "enc2",
    "layers.0.6": "enc3",
    "layers.0.7": "enc4",
    "layers.3":   "bottleneck",
    "layers.4":   "dec4",
    "layers.5":   "dec3",
    "layers.6":   "dec2",
    "layers.7":   "dec1",
}

FMAP_META = {
    "enc1":       ("Encoder 1",  "Low-level edges and textures"),
    "enc2":       ("Encoder 2",  "Shapes and local structure"),
    "enc3":       ("Encoder 3",  "Object parts and semantic regions"),
    "enc4":       ("Encoder 4",  "High-level semantics"),
    "bottleneck": ("Bottleneck", "Most compressed representation"),
    "dec4":       ("Decoder 4",  "Begins recovering spatial detail"),
    "dec3":       ("Decoder 3",  "Recovering object-level color regions"),
    "dec2":       ("Decoder 2",  "Local color via skip features"),
    "dec1":       ("Decoder 1",  "Full-resolution color prediction"),
}


def _fmap_to_b64_list(tensor: torch.Tensor) -> tuple[list[str], int]:
    fmap  = tensor[0].cpu().numpy()
    total = fmap.shape[0]
    out   = []
    for i in range(min(MAX_FMAP_CH, total)):
        ch  = fmap[i]
        ch  = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        img = Image.fromarray((ch * 255).astype(np.uint8), "L").resize(
            (FMAP_SIZE, FMAP_SIZE), Image.BILINEAR
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out, total


# inference

@torch.no_grad()
def run_colorization(net_G, image_bytes: bytes, uncertainty: bool) -> dict:
    device = next(net_G.parameters()).device
    rgb_np, L_raw, L_tensor = _prepare_L(image_bytes)
    L_tensor = L_tensor.to(device)

    raw_fmaps: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in net_G.named_modules():
        for suffix, label in HOOK_TARGETS.items():
            if name == suffix or name.endswith("." + suffix):
                def _make(k):
                    def _h(_, __, out): raw_fmaps[k] = out.detach().cpu()
                    return _h
                hooks.append(module.register_forward_hook(_make(label)))
                break

    try:
        if uncertainty:
            ab_t, log_var_t = net_G(L_tensor)
        else:
            ab_t      = net_G(L_tensor)
            log_var_t = None
    finally:
        for h in hooks:
            h.remove()

    ab      = ab_t[0].permute(1, 2, 0).cpu().numpy() * 110.0
    lab     = np.concatenate([L_raw[:, :, None], ab], axis=-1)
    rgb_pred = np.clip(skcolor.lab2rgb(lab), 0, 1)

    L_disp = L_raw / 100.0
    a_disp = (ab[:, :, 0] / 127.0 + 1) / 2
    b_disp = (ab[:, :, 1] / 127.0 + 1) / 2

    unc_heat = unc_over = None
    if log_var_t is not None:
        std  = torch.exp(log_var_t / 2).mean(dim=1)[0].cpu().numpy()
        sn   = (std - std.min()) / (std.max() - std.min() + 1e-8)
        unc_heat = arr_to_b64(_unc_heatmap(sn).astype(np.float32))
        L_rgb    = np.stack([L_disp] * 3, axis=-1)
        unc_over = arr_to_b64(np.clip(0.55 * L_rgb + 0.45 * _unc_heatmap(sn), 0, 1).astype(np.float32))

    fmaps_out = {}
    for key, (label, desc) in FMAP_META.items():
        if key not in raw_fmaps:
            continue
        tensor = raw_fmaps[key]
        channels, total = _fmap_to_b64_list(tensor)
        _, _C, H, W = tensor.shape
        fmaps_out[key] = {
            "channels": channels,
            "total_ch": total,
            "spatial":  f"{H}×{W}",
            "label":    label,
            "desc":     desc,
        }

    return {
        "input_L":   arr_to_b64(L_disp.astype(np.float32)),
        "original":  arr_to_b64(rgb_np.astype(np.float32)),
        "colorized": arr_to_b64(rgb_pred.astype(np.float32)),
        "ab_a":      arr_to_b64(a_disp.astype(np.float32)),
        "ab_b":      arr_to_b64(b_disp.astype(np.float32)),
        "unc_heat":  unc_heat,
        "unc_over":  unc_over,
        "has_unc":   uncertainty and log_var_t is not None,
        "fmaps":     fmaps_out,
    }


@torch.no_grad()
def compare_all_models(image_bytes: bytes, models: list[dict]) -> dict:
    _, L_raw, L_tensor = _prepare_L(image_bytes)
    rgb_np = (
        np.array(
            Image.open(io.BytesIO(image_bytes))
            .convert("RGB")
            .resize((IMG_SIZE, IMG_SIZE))
        ).astype(np.float32)
        / 255.0
    )
    results = []
    for m in models:
        try:
            net    = load_model(m["path"], m["uncertainty"])
            device = next(net.parameters()).device
            if m["uncertainty"]:
                ab_t, _ = net(L_tensor.to(device))
            else:
                ab_t = net(L_tensor.to(device))
            ab  = ab_t[0].permute(1, 2, 0).cpu().numpy() * 110.0
            lab = np.concatenate([L_raw[:, :, None], ab], axis=-1)
            rgb = np.clip(skcolor.lab2rgb(lab), 0, 1)
            score = float(np.asarray(ssim(rgb_np, rgb, channel_axis=-1, data_range=1.0)).flat[0])
            results.append({
                "name":  m["name"],
                "path":  m["path"],
                "unc":   m["uncertainty"],
                "img":   arr_to_b64(rgb.astype(np.float32)),
                "ssim":  score,
                "error": None,
            })
        except Exception as exc:
            results.append({
                "name":  m["name"],
                "path":  m["path"],
                "unc":   m["uncertainty"],
                "img":   None,
                "ssim":  -1.0,
                "error": str(exc),
            })
    # sort by SSIM, failed models go last
    results.sort(key=lambda r: r["ssim"], reverse=True)
    return {"original": arr_to_b64(rgb_np), "results": results}


# HTML components

def build_unet_html(fmaps: dict, input_L_b64: str, colorized_b64: str) -> str:
    """Build a self-contained HTML page for the interactive U-Net diagram."""
    fmaps_js = json.dumps(fmaps)
    in_src   = b64_src(input_L_b64)
    out_src  = b64_src(colorized_b64)

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#05080f;--surface:rgba(255,255,255,0.04);--border:rgba(255,255,255,0.08);
  --accent:#4f8ef7;--accent2:#7dd8f8;--text:#e8edf8;--muted:#7a8aaa;--radius:14px;
}}
html,body{{background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
body{{padding:.5rem 1rem 1.5rem;overflow-x:hidden;}}
.section-desc{{color:var(--muted);font-size:.88rem;line-height:1.65;margin-bottom:1rem;}}
.hidden{{display:none!important}}

/* ── U-Net diagram ── */
.unet-diagram{{
  position:relative;display:flex;flex-direction:column;align-items:center;
  padding:.5rem 0 1rem;user-select:none;
}}
.unet-svg{{
  position:absolute;inset:0;width:100%;height:100%;
  pointer-events:none;overflow:visible;z-index:1;
}}
.unet-top-row{{display:flex;justify-content:space-between;width:100%;padding:0 0 .5rem;}}
.unet-main{{display:grid;grid-template-columns:1fr 160px 1fr;width:100%;}}
.unet-enc-col,.unet-dec-col{{display:flex;flex-direction:column;gap:48px;padding:.5rem 0;}}
.unet-bot-row{{display:flex;justify-content:center;padding-top:2rem;}}

.ublock{{
  border-radius:10px;padding:.6rem .85rem;display:flex;flex-direction:column;gap:2px;
  cursor:pointer;transition:transform .18s,box-shadow .18s,border-color .18s;
  position:relative;z-index:2;
}}
.ublock:hover{{transform:translateY(-2px);}}
.ublock-enc{{background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.3);}}
.ublock-enc:hover{{box-shadow:0 0 18px rgba(79,142,247,.25);border-color:var(--accent);}}
.ublock-enc.active{{border-color:var(--accent);box-shadow:0 0 22px rgba(79,142,247,.4);}}
.ublock-dec{{background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.3);}}
.ublock-dec:hover{{box-shadow:0 0 18px rgba(99,102,241,.25);border-color:#818cf8;}}
.ublock-dec.active{{border-color:#818cf8;box-shadow:0 0 22px rgba(99,102,241,.4);}}
.ublock-bottleneck{{
  background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.3);
  min-width:200px;text-align:center;align-items:center;
}}
.ublock-bottleneck:hover{{box-shadow:0 0 22px rgba(34,211,238,.25);border-color:#22d3ee;}}
.ublock-bottleneck.active{{border-color:#22d3ee;box-shadow:0 0 28px rgba(34,211,238,.4);}}
.ublock-io{{
  background:rgba(255,255,255,.04);border:1px solid var(--border);
  align-items:center;cursor:default;min-width:110px;
}}
.ublock-thumb{{width:68px;height:68px;border-radius:6px;object-fit:cover;background:#000;display:block;margin-bottom:4px;}}
.ublock-label{{font-size:.8rem;font-weight:600;}}
.ublock-sub  {{font-size:.7rem;color:var(--muted);}}
.ublock-cta  {{font-size:.65rem;color:var(--accent2);margin-top:2px;opacity:.7;}}

/* ── info panel ── */
.info-panel{{
  margin-top:1rem;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);backdrop-filter:blur(16px);animation:slideUp .2s ease;overflow:hidden;
}}
@keyframes slideUp{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
.info-inner{{padding:1.1rem 1.3rem;position:relative;}}
.info-close{{
  position:absolute;top:.75rem;right:.85rem;background:none;border:none;
  color:var(--muted);font-size:1.3rem;cursor:pointer;line-height:1;
}}
.info-close:hover{{color:var(--text);}}
.info-title{{font-size:.95rem;font-weight:700;margin-bottom:.75rem;color:var(--accent2);}}
.info-visual{{
  margin-bottom:.85rem;font-family:monospace;font-size:.8rem;color:var(--muted);
  background:rgba(0,0,0,.3);border-radius:8px;padding:.75rem 1rem;
  line-height:1.8;white-space:pre;overflow-x:auto;
}}
.info-desc{{font-size:.86rem;color:var(--muted);line-height:1.65;}}

/* ── feature-map modal ── */
.fmap-overlay{{
  position:fixed;inset:0;background:rgba(0,0,0,.65);backdrop-filter:blur(6px);
  display:flex;align-items:center;justify-content:center;z-index:200;
  animation:fadeIn .18s ease;
}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
.fmap-modal{{
  background:rgba(7,11,22,.95);border:1px solid rgba(255,255,255,.1);
  border-radius:18px;backdrop-filter:blur(24px);
  width:min(92vw,680px);max-height:88vh;overflow:hidden;
  animation:slideUp .22s ease;box-shadow:0 28px 80px rgba(0,0,0,.75);
  display:flex;flex-direction:column;
}}
.fmap-header{{
  display:flex;align-items:center;gap:1rem;padding:1rem 1.3rem;
  border-bottom:1px solid var(--border);flex-shrink:0;
}}
.fmap-titles{{flex:1;min-width:0;}}
.fmap-title{{font-size:.95rem;font-weight:700;color:var(--accent2);}}
.fmap-desc {{font-size:.76rem;color:var(--muted);margin-top:2px;}}
.fmap-stats{{font-size:.72rem;color:var(--muted);white-space:nowrap;}}
.fmap-close{{
  background:none;border:1px solid var(--border);color:var(--muted);
  width:30px;height:30px;border-radius:50%;cursor:pointer;font-size:1.1rem;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  transition:color .15s,border-color .15s;
}}
.fmap-close:hover{{color:var(--text);border-color:rgba(255,255,255,.25);}}
.fmap-body{{overflow-y:auto;flex:1;}}
.fmap-body::-webkit-scrollbar{{width:4px;}}
.fmap-body::-webkit-scrollbar-thumb{{background:var(--border);border-radius:4px;}}

/* ── deck of cards ── */
.deck-scene{{
  display:flex;flex-direction:column;align-items:center;
  padding:3rem 2rem 5rem;min-height:300px;cursor:crosshair;user-select:none;
}}
.deck{{position:relative;width:120px;height:120px;}}
.deck-card{{
  position:absolute;left:0;top:0;width:120px;height:120px;
  border-radius:9px;overflow:hidden;border:1px solid rgba(255,255,255,.1);
  background:#0c0e18;cursor:pointer;
  transition:box-shadow .25s ease,border-color .2s ease;
  box-shadow:2px 4px 14px rgba(0,0,0,.65);
}}
.deck-card:hover{{border-color:var(--accent);box-shadow:0 0 20px rgba(79,142,247,.6);z-index:100;}}
.deck-card img{{width:100%;height:100%;object-fit:cover;display:block;image-rendering:pixelated;}}
.deck-hint{{
  margin-top:2rem;font-size:.71rem;color:var(--muted);opacity:.6;
  text-align:center;pointer-events:none;line-height:1.7;
}}

/* ── lightbox ── */
.lightbox{{
  position:fixed;inset:0;background:rgba(0,0,0,.9);
  display:flex;align-items:center;justify-content:center;z-index:999;
  animation:fadeIn .15s ease;
}}
.lightbox img{{
  max-width:80vw;max-height:80vh;border-radius:8px;
  image-rendering:pixelated;box-shadow:0 24px 80px rgba(0,0,0,.8);
}}
.lb-nav{{
  position:absolute;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
  color:#fff;font-size:1.4rem;width:44px;height:44px;border-radius:50%;cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:background .15s;
}}
.lb-nav:hover{{background:rgba(255,255,255,.15);}}
#lb-prev{{left:2rem;}}#lb-next{{right:2rem;}}
.lb-counter{{position:absolute;bottom:1.5rem;font-size:.8rem;color:rgba(255,255,255,.5);}}
</style>
</head>
<body>
<p class="section-desc">
  Click any <span style="color:var(--accent)">encoder</span> or
  <span style="color:#818cf8">decoder</span> block to explore its feature maps.
  Click the arrows to understand each layer operation.
</p>

<div id="unet-diagram" class="unet-diagram">
  <svg id="unet-svg" class="unet-svg"></svg>

  <div class="unet-top-row">
    <div class="ublock ublock-io" id="ub-input">
      <img class="ublock-thumb" src="{in_src}" />
      <span class="ublock-label">Input</span>
      <span class="ublock-sub">L &middot; 256&times;256</span>
    </div>
    <div class="ublock ublock-io" id="ub-output">
      <img class="ublock-thumb" src="{out_src}" />
      <span class="ublock-label">Output ab</span>
      <span class="ublock-sub">2 ch &middot; 256&times;256</span>
    </div>
  </div>

  <div class="unet-main">
    <div class="unet-enc-col">
      <div class="ublock ublock-enc" id="ub-enc1" data-key="enc1">
        <span class="ublock-label">Encoder 1</span>
        <span class="ublock-sub">64 ch &middot; 128&times;128</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-enc" id="ub-enc2" data-key="enc2">
        <span class="ublock-label">Encoder 2</span>
        <span class="ublock-sub">128 ch &middot; 64&times;64</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-enc" id="ub-enc3" data-key="enc3">
        <span class="ublock-label">Encoder 3</span>
        <span class="ublock-sub">256 ch &middot; 32&times;32</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-enc" id="ub-enc4" data-key="enc4">
        <span class="ublock-label">Encoder 4</span>
        <span class="ublock-sub">512 ch &middot; 16&times;16</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
    </div>

    <div id="unet-center-col"></div>

    <div class="unet-dec-col">
      <div class="ublock ublock-dec" id="ub-dec1" data-key="dec1">
        <span class="ublock-label">Decoder 1</span>
        <span class="ublock-sub">64 ch &middot; 128&times;128</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-dec" id="ub-dec2" data-key="dec2">
        <span class="ublock-label">Decoder 2</span>
        <span class="ublock-sub">128 ch &middot; 64&times;64</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-dec" id="ub-dec3" data-key="dec3">
        <span class="ublock-label">Decoder 3</span>
        <span class="ublock-sub">256 ch &middot; 32&times;32</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
      <div class="ublock ublock-dec" id="ub-dec4" data-key="dec4">
        <span class="ublock-label">Decoder 4</span>
        <span class="ublock-sub">512 ch &middot; 16&times;16</span>
        <span class="ublock-cta">tap &middot; feature maps</span>
      </div>
    </div>
  </div>

  <div class="unet-bot-row">
    <div class="ublock ublock-bottleneck" id="ub-bottleneck" data-key="bottleneck">
      <span class="ublock-label">Bottleneck</span>
      <span class="ublock-sub">512 ch &middot; 8&times;8</span>
      <span class="ublock-cta">tap &middot; feature maps</span>
    </div>
  </div>
</div>

<div id="info-panel" class="info-panel hidden">
  <div class="info-inner">
    <button id="info-close" class="info-close">&times;</button>
    <div id="info-title" class="info-title"></div>
    <div id="info-visual" class="info-visual"></div>
    <div id="info-desc"  class="info-desc"></div>
  </div>
</div>

<div id="lightbox" class="lightbox hidden">
  <button id="lb-prev" class="lb-nav">&#8592;</button>
  <img id="lightbox-img" alt="feature map" />
  <button id="lb-next" class="lb-nav">&#8594;</button>
  <span id="lb-counter" class="lb-counter"></span>
</div>

<script>
// ── pre-loaded data ──────────────────────────────────────────────────────────
const FEATURE_MAPS = {fmaps_js};

const OP_INFO = {{
  downsample: {{
    title: "Downsampling — Strided Convolution",
    visual: "Input:  [B, C,  H,  W]\\nConv2d: kernel=3, stride=2\\nOutput: [B, 2C, H/2, W/2]\\n\\n  stride 2 samples every other position\\n  spatial size halved, channels doubled",
    desc:  "Each encoder step applies a strided convolution that halves the spatial resolution (H×W → H/2×W/2) while doubling channels. This forces the network to compress spatial information into richer feature representations."
  }},
  upsample: {{
    title: "Upsampling — PixelShuffle",
    visual: "Input:  [B, C×r², H,   W  ]   r=2\\nOutput: [B, C,    H×2, W×2]\\n\\n  Low-res many channels → High-res fewer\\n  Sub-pixel conv avoids checkerboard artifacts",
    desc:  "PixelShuffle rearranges elements from a low-resolution feature map with many channels into a high-resolution map with fewer channels. This avoids the checkerboard artifacts produced by transposed convolutions."
  }},
  skip: {{
    title: "Skip Connection — Concatenation",
    visual: "  Encoder feat  ───────────────────┐\\n                                    cat()\\n  Decoder feat  →  [ enc_feat | dec_feat ]\\n                     channels are doubled",
    desc:  "Skip connections copy encoder feature maps directly to the corresponding decoder level and concatenate them channel-wise. This gives the decoder direct access to fine-grained spatial detail that would otherwise be lost through the bottleneck."
  }},
  bottleneck_down: {{
    title: "Encoder → Bottleneck",
    visual: "  Encoder 4:   [B, 512, 16, 16]\\n  Conv 3×3 ×2\\n  Bottleneck:  [B, 512,  8,  8]\\n\\n  Most compressed — 8×8 covers full scene",
    desc:  "The final downsampling step produces the bottleneck. At 8×8 resolution, each position covers a 32×32 receptive field in the original input. This is where the network decides what color to use based on high-level scene understanding."
  }},
  bottleneck_up: {{
    title: "Bottleneck → Decoder",
    visual: "  Bottleneck: [B, 512,  8,  8]\\n  PixelShuffle r=2\\n  Decoder 4:  [B, 512, 16, 16]\\n  + skip from Encoder 4\\n  concat:     [B,1024, 16, 16]",
    desc:  "The bottleneck output is upsampled to 16×16 and merged with the skip connection from Encoder 4. High-level semantic context combines with the spatial detail preserved in the skip connection."
  }},
  output_conv: {{
    title: "Output Projection — Conv 1×1 + Tanh",
    visual: "  Decoder 1: [B,  64, 128, 128]\\n  Conv 1×1\\n  Output:    [B,   2, 256, 256]\\n  Tanh → values in [−1, 1]\\n\\n  ch 0 → a (green/red axis)\\n  ch 1 → b (blue/yellow axis)",
    desc:  "A 1×1 convolution collapses 64 channels to exactly 2 — the predicted a and b color channels. Tanh squashes to [−1,1]. These two channels are rescaled by 110 and merged with the original L channel to produce the final RGB image."
  }},
}};

// ── lightbox state ──────────────────────────────────────────────────────────
let lbImages = [], lbIdx = 0;
const lightbox = document.getElementById("lightbox");
const lbImg    = document.getElementById("lightbox-img");
const lbCtr    = document.getElementById("lb-counter");

// ── init ────────────────────────────────────────────────────────────────────
document.querySelectorAll(".ublock[data-key]").forEach(el => {{
  if (!FEATURE_MAPS[el.dataset.key]) {{
    el.querySelector(".ublock-cta") && (el.querySelector(".ublock-cta").style.opacity = "0.3");
    el.style.cursor = "default";
  }} else {{
    el.addEventListener("click", () => openFmapPanel(el.dataset.key, el));
  }}
}});

requestAnimationFrame(() => requestAnimationFrame(drawUnetSvg));
window.addEventListener("resize", drawUnetSvg);

// ── SVG helpers ─────────────────────────────────────────────────────────────
const _mkrs = new Set();
function ensureMarker(svg, color) {{
  const id = "arr-" + color.replace("#","");
  if (_mkrs.has(id)) return; _mkrs.add(id);
  let defs = svg.querySelector("defs");
  if (!defs) {{ defs = document.createElementNS("http://www.w3.org/2000/svg","defs"); svg.prepend(defs); }}
  const mk = document.createElementNS("http://www.w3.org/2000/svg","marker");
  mk.setAttribute("id",id); mk.setAttribute("viewBox","0 0 10 10");
  mk.setAttribute("refX","8"); mk.setAttribute("refY","5");
  mk.setAttribute("markerWidth","7"); mk.setAttribute("markerHeight","7");
  mk.setAttribute("orient","auto");
  const p = document.createElementNS("http://www.w3.org/2000/svg","path");
  p.setAttribute("d","M 0 0 L 10 5 L 0 10 z"); p.setAttribute("fill",color);
  mk.appendChild(p); defs.appendChild(mk);
}}

function drawArrow(svg, x1, y1, x2, y2, color, opKey) {{
  const g = document.createElementNS("http://www.w3.org/2000/svg","g");
  g.style.pointerEvents = "all";

  const ln = document.createElementNS("http://www.w3.org/2000/svg","line");
  ln.setAttribute("x1",x1); ln.setAttribute("y1",y1);
  ln.setAttribute("x2",x2); ln.setAttribute("y2",y2);
  ln.setAttribute("stroke",color); ln.setAttribute("stroke-width","2.5");
  ln.setAttribute("stroke-opacity","0.6");
  ln.setAttribute("marker-end",`url(#arr-${{color.replace("#","")}})`);
  ln.style.pointerEvents = "none";

  const hit = document.createElementNS("http://www.w3.org/2000/svg","line");
  hit.setAttribute("x1",x1); hit.setAttribute("y1",y1);
  hit.setAttribute("x2",x2); hit.setAttribute("y2",y2);
  hit.setAttribute("stroke","transparent"); hit.setAttribute("stroke-width","28");
  hit.style.cursor = "pointer"; hit.style.pointerEvents = "all";
  hit.addEventListener("click", e => {{ e.stopPropagation(); showInfoPanel(opKey); }});
  hit.addEventListener("mouseenter", () => {{ ln.setAttribute("stroke-opacity","1"); ln.setAttribute("stroke-width","3.5"); }});
  hit.addEventListener("mouseleave", () => {{ ln.setAttribute("stroke-opacity","0.6"); ln.setAttribute("stroke-width","2.5"); }});

  const mx=(x1+x2)/2, my=(y1+y2)/2;
  const dx=y2-y1, dy=-(x2-x1), len=Math.sqrt(dx*dx+dy*dy)||1;
  const label = OP_INFO[opKey] ? OP_INFO[opKey].title.split("—")[0].trim() : "";
  if (label) {{
    const tx = document.createElementNS("http://www.w3.org/2000/svg","text");
    tx.setAttribute("x", mx+(dx/len)*10); tx.setAttribute("y", my+(dy/len)*10);
    tx.setAttribute("fill",color); tx.setAttribute("font-size","9");
    tx.setAttribute("opacity","0.55"); tx.setAttribute("text-anchor","middle");
    tx.setAttribute("dominant-baseline","middle"); tx.style.pointerEvents="none";
    tx.textContent=label; g.appendChild(tx);
  }}

  ensureMarker(svg,color); g.appendChild(ln); g.appendChild(hit); svg.appendChild(g);
}}

function drawSkip(svg, x1, y1, x2, y2) {{
  const g = document.createElementNS("http://www.w3.org/2000/svg","g");
  g.style.pointerEvents = "all";

  const ln = document.createElementNS("http://www.w3.org/2000/svg","line");
  ln.setAttribute("x1",x1); ln.setAttribute("y1",y1);
  ln.setAttribute("x2",x2); ln.setAttribute("y2",y2);
  ln.setAttribute("stroke","#f0a0c8"); ln.setAttribute("stroke-width","2");
  ln.setAttribute("stroke-opacity","0.55"); ln.setAttribute("stroke-dasharray","6 4");
  ln.setAttribute("marker-end","url(#arr-f0a0c8)"); ln.style.pointerEvents="none";

  const hit = document.createElementNS("http://www.w3.org/2000/svg","line");
  hit.setAttribute("x1",x1); hit.setAttribute("y1",y1);
  hit.setAttribute("x2",x2); hit.setAttribute("y2",y2);
  hit.setAttribute("stroke","transparent"); hit.setAttribute("stroke-width","24");
  hit.style.cursor="pointer"; hit.style.pointerEvents="all";
  hit.addEventListener("click", e => {{ e.stopPropagation(); showInfoPanel("skip"); }});
  hit.addEventListener("mouseenter", () => {{ ln.setAttribute("stroke-opacity","1"); ln.setAttribute("stroke-width","3"); }});
  hit.addEventListener("mouseleave", () => {{ ln.setAttribute("stroke-opacity","0.55"); ln.setAttribute("stroke-width","2"); }});

  const mx=(x1+x2)/2, my=(y1+y2)/2;
  const tx = document.createElementNS("http://www.w3.org/2000/svg","text");
  tx.setAttribute("x",mx); tx.setAttribute("y",my-8);
  tx.setAttribute("fill","#f0a0c8"); tx.setAttribute("font-size","9");
  tx.setAttribute("opacity","0.55"); tx.setAttribute("text-anchor","middle");
  tx.style.pointerEvents="none"; tx.textContent="skip"; g.appendChild(tx);

  ensureMarker(svg,"#f0a0c8"); g.appendChild(ln); g.appendChild(hit); svg.appendChild(g);
}}

function drawUnetSvg() {{
  const svg  = document.getElementById("unet-svg");
  const diag = document.getElementById("unet-diagram");
  svg.innerHTML=""; _mkrs.clear();
  svg.style.pointerEvents="none";
  const dr = diag.getBoundingClientRect();
  function pos(id) {{
    const r = document.getElementById(id).getBoundingClientRect();
    return {{cx:r.left-dr.left+r.width/2, cy:r.top-dr.top+r.height/2,
             x:r.left-dr.left, r:r.right-dr.left, y:r.top-dr.top, b:r.bottom-dr.top}};
  }}
  const encIds=["ub-enc1","ub-enc2","ub-enc3","ub-enc4"];
  const decIds=["ub-dec1","ub-dec2","ub-dec3","ub-dec4"];
  for (let i=0;i<encIds.length-1;i++) {{
    const a=pos(encIds[i]),b=pos(encIds[i+1]);
    drawArrow(svg,a.cx,a.b,b.cx,b.y,"#4f8ef7","downsample");
  }}
  const enc4=pos("ub-enc4"),bot=pos("ub-bottleneck");
  drawArrow(svg,enc4.cx,enc4.b,bot.cx,bot.y,"#22d3ee","bottleneck_down");
  const dec4=pos("ub-dec4");
  drawArrow(svg,bot.cx,bot.y,dec4.cx,dec4.b,"#818cf8","bottleneck_up");
  for (let i=decIds.length-1;i>0;i--) {{
    const a=pos(decIds[i]),b=pos(decIds[i-1]);
    drawArrow(svg,a.cx,a.y,b.cx,b.b,"#818cf8","upsample");
  }}
  [["ub-enc1","ub-dec1"],["ub-enc2","ub-dec2"],["ub-enc3","ub-dec3"],["ub-enc4","ub-dec4"]]
    .forEach(([eid,did]) => {{
      const e=pos(eid),d=pos(did); drawSkip(svg,e.r,e.cy,d.x,d.cy);
    }});
  const dec1=pos("ub-dec1"),outp=pos("ub-output");
  drawArrow(svg,dec1.cx,dec1.y,outp.cx,outp.b,"#22d3ee","output_conv");
  const inp=pos("ub-input"),enc1=pos("ub-enc1");
  drawArrow(svg,inp.cx,inp.b,enc1.cx,enc1.y,"#4f8ef7","downsample");
}}

// ── info panel ──────────────────────────────────────────────────────────────
function showInfoPanel(opKey) {{
  const info = OP_INFO[opKey]; if (!info) return;
  closeFmapPanel();
  document.getElementById("info-title").textContent  = info.title;
  document.getElementById("info-visual").textContent = info.visual;
  document.getElementById("info-desc").textContent   = info.desc;
  const p = document.getElementById("info-panel");
  p.classList.remove("hidden");
  p.scrollIntoView({{behavior:"smooth",block:"nearest"}});
}}
document.getElementById("info-close").addEventListener("click", () => {{
  document.getElementById("info-panel").classList.add("hidden");
}});

// ── deck modal ───────────────────────────────────────────────────────────────
function cardTransform(card, ni, spread) {{
  const rotY=-27+spread*18, rotX=11-spread*8;
  const x=(ni-15)*(1-spread)*.9+(ni-7.5)*spread*52;
  const y=(15-ni)*(1-spread)*.9;
  const z=(ni-15)*(1-spread)*10;
  card.style.transform =
    `perspective(700px) rotateY(${{rotY.toFixed(2)}}deg) rotateX(${{rotX.toFixed(2)}}deg)` +
    ` translateX(${{x.toFixed(1)}}px) translateY(${{y.toFixed(1)}}px) translateZ(${{z.toFixed(1)}}px)`;
}}

function openFmapPanel(key, blockEl) {{
  const fm = FEATURE_MAPS[key]; if (!fm) return;
  closeFmapPanel();
  document.querySelectorAll(".ublock").forEach(b => b.classList.remove("active"));
  if (blockEl) blockEl.classList.add("active");

  lbImages = fm.channels.map(b64 => "data:image/png;base64," + b64);
  const n=fm.channels.length, maxI=Math.max(n-1,1);

  const scene = document.createElement("div"); scene.className="deck-scene";
  const deck  = document.createElement("div"); deck.className="deck";
  const cards = [];

  fm.channels.forEach((ch,i) => {{
    const card = document.createElement("div"); card.className="deck-card";
    const ni=(i/maxI)*15; cardTransform(card,ni,0);
    const img=document.createElement("img");
    img.src="data:image/png;base64,"+ch; img.alt="ch"+i;
    card.appendChild(img);
    card.addEventListener("click", e => {{ e.stopPropagation(); openLightbox(i); }});
    deck.appendChild(card); cards.push({{el:card,ni}});
  }});

  scene.addEventListener("mousemove", e => {{
    const r=scene.getBoundingClientRect();
    const spread=1-Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
    cards.forEach(c => cardTransform(c.el,c.ni,spread));
  }});
  scene.addEventListener("mouseleave", () => cards.forEach(c => cardTransform(c.el,c.ni,0)));

  const hint=document.createElement("p"); hint.className="deck-hint";
  hint.textContent="Slide mouse left to fan · Click any filter to expand";
  scene.appendChild(deck); scene.appendChild(hint);

  const overlay=document.createElement("div"); overlay.className="fmap-overlay";
  const modal  =document.createElement("div"); modal.className="fmap-modal";
  const hdr    =document.createElement("div"); hdr.className="fmap-header";
  const titles =document.createElement("div"); titles.className="fmap-titles";
  titles.innerHTML=`<div class="fmap-title">${{fm.label}}</div><div class="fmap-desc">${{fm.desc}}</div>`;
  const stats =document.createElement("div"); stats.className="fmap-stats";
  stats.textContent=`${{n}} of ${{fm.total_ch}} ch · ${{fm.spatial}}`;
  const closeBtn=document.createElement("button"); closeBtn.className="fmap-close";
  closeBtn.textContent="×"; closeBtn.addEventListener("click",closeFmapPanel);
  hdr.appendChild(titles); hdr.appendChild(stats); hdr.appendChild(closeBtn);
  const body=document.createElement("div"); body.className="fmap-body";
  body.appendChild(scene);
  modal.appendChild(hdr); modal.appendChild(body);
  overlay.appendChild(modal);
  overlay.addEventListener("click", e => {{ if(e.target===overlay) closeFmapPanel(); }});
  document.body.appendChild(overlay);

  // fan animation on open
  let t=0;
  const anim=setInterval(()=>{{
    t+=.05;
    const s=Math.sin(t*Math.PI)*.65;
    cards.forEach(c=>cardTransform(c.el,c.ni,s));
    if(t>=1){{clearInterval(anim);cards.forEach(c=>cardTransform(c.el,c.ni,0));}}
  }},16);
}}

function closeFmapPanel() {{
  const ov=document.querySelector(".fmap-overlay"); if(ov) ov.remove();
  document.querySelectorAll(".ublock").forEach(b=>b.classList.remove("active"));
}}

// ── lightbox ─────────────────────────────────────────────────────────────────
function openLightbox(idx) {{
  lbIdx=idx; updateLb(); lightbox.classList.remove("hidden");
}}
function updateLb() {{
  lbImg.src=lbImages[lbIdx]; lbCtr.textContent=`${{lbIdx+1}} / ${{lbImages.length}}`;
}}
document.getElementById("lb-prev").addEventListener("click",e=>{{
  e.stopPropagation(); lbIdx=(lbIdx-1+lbImages.length)%lbImages.length; updateLb();
}});
document.getElementById("lb-next").addEventListener("click",e=>{{
  e.stopPropagation(); lbIdx=(lbIdx+1)%lbImages.length; updateLb();
}});
lightbox.addEventListener("click",e=>{{
  if(e.target===lightbox||e.target===lbImg) lightbox.classList.add("hidden");
}});
document.addEventListener("keydown",e=>{{
  if(!lightbox.classList.contains("hidden")){{
    if(e.key==="ArrowLeft") {{lbIdx=(lbIdx-1+lbImages.length)%lbImages.length;updateLb();}}
    if(e.key==="ArrowRight"){{lbIdx=(lbIdx+1)%lbImages.length;updateLb();}}
    if(e.key==="Escape")    {{lightbox.classList.add("hidden");}}
  }} else if(e.key==="Escape") closeFmapPanel();
}});
</script>
</body>
</html>"""


def build_slider_html(before_b64: str, after_b64: str, size: int = 280,
                      label_before: str = "Before", label_after: str = "After") -> str:
    """Build a drag-to-compare slider HTML page."""
    gs = b64_src(before_b64)
    cs = b64_src(after_b64)
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#05080f;display:flex;flex-direction:column;align-items:center;padding:.75rem;}}
h3{{color:#e8edf8;font-family:-apple-system,sans-serif;font-size:.85rem;font-weight:600;
    margin-bottom:.6rem;opacity:.7;letter-spacing:.05em;text-transform:uppercase;}}
.wrap{{
  position:relative;width:{size}px;height:{size}px;
  border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.1);
  cursor:col-resize;user-select:none;
  box-shadow:0 8px 32px rgba(0,0,0,.5);
}}
.base{{position:absolute;inset:0;}}
.base img{{width:100%;height:100%;object-fit:cover;display:block;}}
.clip{{position:absolute;top:0;left:0;height:100%;overflow:hidden;width:50%;}}
.clip img{{width:{size}px;height:{size}px;object-fit:cover;display:block;}}
.divider{{
  position:absolute;top:0;left:50%;width:2px;height:100%;
  background:rgba(255,255,255,.7);transform:translateX(-50%);pointer-events:none;
}}
.handle{{
  position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  width:38px;height:38px;background:rgba(10,14,28,.85);
  border:2px solid rgba(255,255,255,.55);border-radius:50%;
  display:flex;align-items:center;justify-content:center;gap:1px;
  pointer-events:none;backdrop-filter:blur(4px);
}}
.badge{{
  position:absolute;top:.55rem;font-size:.65rem;font-weight:700;
  color:white;background:rgba(0,0,0,.55);padding:.12rem .4rem;
  border-radius:4px;letter-spacing:.04em;text-transform:uppercase;
}}
.badge-l{{left:.55rem;}} .badge-r{{right:.55rem;}}
</style>
</head>
<body>
<h3>{label_before} &amp; {label_after}</h3>
<div class="wrap" id="wrap">
  <div class="base"><img src="{cs}" alt="after" /></div>
  <div class="clip" id="clip">
    <img src="{gs}" alt="before" />
    <span class="badge badge-l">{label_before}</span>
  </div>
  <div class="divider" id="divider"></div>
  <div class="handle" id="handle">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
      <polyline points="15 18 9 12 15 6"></polyline>
    </svg>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
      <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
  </div>
  <span class="badge badge-r">{label_after}</span>
</div>
<script>
const wrap=document.getElementById("wrap");
const clip=document.getElementById("clip");
const div =document.getElementById("divider");
const hdl =document.getElementById("handle");
let drag=false;
function setPos(e){{
  const r=wrap.getBoundingClientRect();
  const cx=e.touches?e.touches[0].clientX:e.clientX;
  const pct=(Math.max(0,Math.min(r.width,cx-r.left))/r.width*100).toFixed(1);
  clip.style.width=pct+"%";
  div.style.left=pct+"%";
  hdl.style.left=pct+"%";
}}
wrap.addEventListener("mousedown",e=>{{drag=true;setPos(e);}});
document.addEventListener("mousemove",e=>{{if(drag)setPos(e);}});
document.addEventListener("mouseup",()=>{{drag=false;}});
wrap.addEventListener("touchstart",e=>setPos(e),{{passive:true}});
wrap.addEventListener("touchmove",e=>{{setPos(e);e.preventDefault();}},{{passive:false}});
</script>
</body>
</html>"""


# misc helpers

def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


def _safe_stem(name: str) -> str:
    """Strip extension and replace unsafe filename characters."""
    stem = os.path.splitext(name)[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


# CSS injection

def inject_css() -> None:
    st.markdown("""
<style>
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* gradient headline */
.col-headline {
  font-size: 2rem; font-weight: 700; letter-spacing: -.5px;
  background: linear-gradient(120deg, #93c5fd, #e0e7ff, #67e8f9);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.2; margin-bottom: .3rem;
}
.col-subtitle {
  color: #7a8aaa; font-size: .93rem; line-height: 1.55; margin-bottom: 1.5rem;
}

/* section headers */
.step-header {
  font-size: 1.1rem; font-weight: 700; color: #e8edf8;
  margin-bottom: .35rem; letter-spacing: -.15px;
}
.step-desc {
  color: #7a8aaa; font-size: .87rem; line-height: 1.6; margin-bottom: 1rem;
}

/* card hover lift */
[data-testid="stImage"] {
  transition: transform .2s, box-shadow .2s;
}
[data-testid="stImage"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(0,0,0,.35);
}

/* colorize button */
div[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #4f8ef7, #6366f1) !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(79,142,247,.35) !important;
  font-weight: 600 !important; letter-spacing: .01em;
  transition: opacity .15s, box-shadow .15s, transform .1s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
  opacity: .9 !important;
  box-shadow: 0 6px 24px rgba(79,142,247,.5) !important;
  transform: translateY(-1px) !important;
}

/* uncertainty badge in sidebar */
.unc-badge {
  display: inline-block; font-size: .73rem; font-weight: 700;
  padding: .2rem .65rem; border-radius: 20px;
  background: rgba(34,211,238,.12); border: 1px solid rgba(34,211,238,.3);
  color: #22d3ee;
}
</style>
""", unsafe_allow_html=True)


# sidebar

def render_sidebar(models: list[dict]) -> dict:
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = set()
    favs: set = st.session_state["favorites"]

    with st.sidebar:
        # model selector
        st.markdown("### Model")

        def _fmt(i: int) -> str:
            m   = models[i]
            pre = "⭐ " if m["path"] in favs else ""
            suf = "  · uncertainty" if m["uncertainty"] else ""
            return f"{pre}{m['name']}{suf}"

        idx = st.selectbox(
            "Checkpoint",
            range(len(models)),
            format_func=_fmt,
            label_visibility="collapsed",
            key="model_idx",
        )
        sel = models[st.session_state.get("model_idx", 0)]

        # pin / unpin button
        is_fav  = sel["path"] in favs
        pin_lbl = "⭐ Unpin" if is_fav else "☆ Pin this model"
        if st.button(pin_lbl, use_container_width=True):
            if is_fav:
                favs.discard(sel["path"])
            else:
                favs.add(sel["path"])
            st.rerun()

        st.markdown("---")
        st.markdown(f"**{sel['name']}**")
        if sel["uncertainty"]:
            st.markdown('<span class="unc-badge">uncertainty head</span>', unsafe_allow_html=True)
        else:
            st.caption("Base generator")

        # upload checkpoint
        st.markdown("---")
        st.markdown("### Upload checkpoint")
        up_model = st.file_uploader(
            "Drop a .pth or .pt file",
            type=["pth", "pt"],
            label_visibility="collapsed",
            help="The file will be saved to the models/ folder and immediately available.",
        )
        if up_model is not None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            dest = os.path.join(MODELS_DIR, up_model.name)
            with open(dest, "wb") as f:
                f.write(up_model.getvalue())
            get_available_models.clear()
            st.success(f"Saved → {up_model.name}")
            st.rerun()

        # footer
        st.markdown("---")
        device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.caption(f"Running on: **{device_label}**")

    return sel


# tab renderers

def render_results(result: dict, model_name: str = "model") -> None:
    stem = _safe_stem(model_name)

    st.markdown('<p class="step-header">Step 1 — LAB color space</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="step-desc">The model predicts only the <b>a</b> and <b>b</b> color channels in LAB space. '
        'The <b>L</b> (luminance) channel is passed through unchanged — it is exactly the grayscale input.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.image(b64_src(result["input_L"]),   caption="L channel (input)",         use_container_width=True)
    c2.image(b64_src(result["ab_a"]),      caption="Predicted a (green↔red)",   use_container_width=True)
    c3.image(b64_src(result["ab_b"]),      caption="Predicted b (blue↔yellow)", use_container_width=True)
    c4.image(b64_src(result["colorized"]), caption="Final colorization",        use_container_width=True)
    # download row
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download L",         _b64_to_bytes(result["input_L"]),   f"{stem}_L.png",         "image/png", use_container_width=True, key="dl_L")
    d2.download_button("Download a",         _b64_to_bytes(result["ab_a"]),      f"{stem}_a.png",         "image/png", use_container_width=True, key="dl_a")
    d3.download_button("Download b",         _b64_to_bytes(result["ab_b"]),      f"{stem}_b.png",         "image/png", use_container_width=True, key="dl_b")
    d4.download_button("Download colorized", _b64_to_bytes(result["colorized"]), f"{stem}_colorized.png", "image/png", use_container_width=True, key="dl_colorized_lab")

    st.markdown("---")
    st.markdown('<p class="step-header">Step 2 — Grayscale vs Colorization</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="step-desc">Drag the handle to compare the grayscale input and the model\'s colorization.</p>',
        unsafe_allow_html=True,
    )
    sl1, sl2 = st.columns(2)
    with sl1:
        components.html(
            build_slider_html(
                result["input_L"], result["colorized"], size=300,
                label_before="Grayscale", label_after="Colorized",
            ),
            height=370,
        )
    with sl2:
        components.html(
            build_slider_html(
                result["original"], result["colorized"], size=300,
                label_before="Original", label_after="Colorized",
            ),
            height=370,
        )

    st.markdown("---")
    st.markdown('<p class="step-header">Step 3 — Final result</p>', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    r1.image(b64_src(result["original"]),  caption="Original (color reference)", use_container_width=True)
    r2.image(b64_src(result["colorized"]), caption="Model colorization",         use_container_width=True)
    dl1, dl2 = st.columns(2)
    dl1.download_button("Download original",  _b64_to_bytes(result["original"]),  f"{stem}_original.png",  "image/png", use_container_width=True, key="dl_original")
    dl2.download_button("Download colorized", _b64_to_bytes(result["colorized"]), f"{stem}_colorized.png", "image/png", use_container_width=True, key="dl_colorized_final")


def render_unet(result: dict) -> None:
    fmaps = result.get("fmaps", {})
    if fmaps:
        n_hooked = len(fmaps)
        st.success(
            f"Feature maps captured from {n_hooked} layers. "
            "Click any block to explore what each layer 'sees'."
        )
    else:
        st.warning(
            "No feature maps were captured — the model architecture may use different layer names. "
            "The diagram and operation descriptions still work."
        )

    components.html(
        build_unet_html(fmaps, result["input_L"], result["colorized"]),
        height=980,
        scrolling=False,
    )


def render_uncertainty(result: dict) -> None:
    if not result.get("has_unc"):
        st.info(
            "This checkpoint does not have an uncertainty head. "
            "Select a model whose filename contains 'uncertainty' to see this visualization."
        )
        return

    st.markdown('<p class="step-header">Where is the model uncertain?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="step-desc">'
        "The model predicts not just color but also its confidence. "
        "<span style='color:#ef4444'>Red</span> = high uncertainty (walls, metal, ambiguous areas). "
        "<span style='color:#3b82f6'>Blue</span> = high confidence (sky, grass, skin)."
        "</p>",
        unsafe_allow_html=True,
    )
    u1, u2 = st.columns(2)
    u1.image(b64_src(result["unc_heat"]), caption="Uncertainty heatmap",          use_container_width=True)
    u2.image(b64_src(result["unc_over"]), caption="Uncertainty overlay on input", use_container_width=True)


def render_comparison(models: list[dict], image_bytes: bytes) -> None:
    st.markdown('<p class="step-header">All models — ranked by similarity</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="step-desc">'
        "Models are ranked by SSIM (Structural Similarity Index) against the original color photo — "
        "higher is more similar to the reference."
        "</p>",
        unsafe_allow_html=True,
    )

    if st.button("Run comparison", type="primary"):
        with st.spinner("Running inference with all checkpoints…"):
            comp = compare_all_models(image_bytes, models)
        st.session_state["comparison"] = comp

    if "comparison" not in st.session_state:
        st.caption("Click the button above to compare all checkpoints.")
        return

    comp     = st.session_state["comparison"]
    results  = comp["results"]
    medals   = ["🥇", "🥈", "🥉"]
    cols_per = 3

    # reference image row
    st.markdown("**Reference**")
    ref_col, *_ = st.columns(cols_per)
    ref_col.image(b64_src(comp["original"]), caption="Original (color reference)", use_container_width=True)
    st.markdown("---")

    # ranked models
    for row_start in range(0, len(results), cols_per):
        chunk = results[row_start : row_start + cols_per]
        cols  = st.columns(cols_per)
        for col, item in zip(cols, chunk):
            rank = results.index(item)
            if item["error"]:
                col.error(f"{item['name']}\n{item['error']}")
                continue
            medal   = medals[rank] if rank < 3 else f"#{rank + 1}"
            unc_tag = "  · uncertainty" if item["unc"] else ""
            caption = f"{medal}  {item['name']}{unc_tag}\nSSIM: {item['ssim']:.4f}"
            col.image(b64_src(item["img"]), caption=caption, use_container_width=True)
            col.download_button(
                "Download",
                _b64_to_bytes(item["img"]),
                file_name=f"{_safe_stem(item['name'])}_colorized.png",
                mime="image/png",
                use_container_width=True,
                key=f"dl_cmp_{rank}",
            )
            if col.button("Use this model", key=f"use_cmp_{rank}", use_container_width=True):
                # find the model index by path
                for i, m in enumerate(models):
                    if m["path"] == item["path"]:
                        st.session_state["pending_model_idx"] = i
                        st.session_state["pending_colorize"]  = True
                        break
                st.rerun()


# main

def main() -> None:
    st.set_page_config(
        page_title="Colorization Explorer",
        page_icon="🎨",
        layout="wide",
    )
    inject_css()

    st.markdown(
        '<h1 class="col-headline">Colorization Explorer</h1>'
        '<p class="col-subtitle">'
        "An interactive look inside a GAN-based grayscale image colorization model — "
        "LAB decomposition, U-Net feature maps, uncertainty visualization, and model comparison."
        "</p>",
        unsafe_allow_html=True,
    )

    models = get_available_models()
    if not models:
        st.warning(
            "No model checkpoints found. Place `.pth` or `.pt` files in `models/` "
            "or set the `MODELS_DIR` environment variable before launching the app."
        )
        return

    # drain pending switch before the selectbox widget is created
    if "pending_model_idx" in st.session_state:
        st.session_state["model_idx"] = st.session_state.pop("pending_model_idx")

    selected = render_sidebar(models)

    try:
        model = load_model(selected["path"], selected["uncertainty"])
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    uploaded = st.file_uploader(
        "Upload a photo",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Works best with natural photos — landscapes, portraits, architecture.",
    )
    if not uploaded:
        st.info("Upload a photo above to begin. The image is resized to 256×256 for the generator.")
        return

    image_bytes = uploaded.read()
    original    = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col_img, col_btn, col_pad = st.columns([2, 1, 2])
    with col_img:
        st.image(original, caption="Uploaded image", width=380)
    with col_btn:
        st.write("")
        clicked = st.button("Colorize", type="primary", use_container_width=True)

    # "Use this model" in comparison sets this flag
    if st.session_state.pop("pending_colorize", False):
        clicked = True

    if clicked:
        with st.spinner("Running inference…"):
            result = run_colorization(model, image_bytes, selected["uncertainty"])
        st.session_state["result"]      = result
        st.session_state["image_bytes"] = image_bytes
        st.session_state.pop("comparison", None)

    if "result" not in st.session_state:
        return

    result = st.session_state["result"]

    st.markdown("---")
    tab_res, tab_unet, tab_unc, tab_cmp = st.tabs([
        "Results",
        "Network Explorer",
        "Uncertainty",
        "Model Comparison",
    ])

    with tab_res:
        render_results(result, model_name=selected["name"])

    with tab_unet:
        render_unet(result)

    with tab_unc:
        render_uncertainty(result)

    with tab_cmp:
        render_comparison(models, st.session_state["image_bytes"])

    st.markdown("---")
    st.caption(
        "Bachelor thesis — Faculty of Automation, Computers and Electronics, University of Craiova · "
        "GAN-based grayscale image colorization with Spatial Color Affinity and No Grey losses"
    )


if __name__ == "__main__":
    main()
