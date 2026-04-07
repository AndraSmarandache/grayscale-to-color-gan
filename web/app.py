import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import color as skcolor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.generator import build_res_unet, UncertaintyGenerator

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(ROOT, "web", "model.pt"))
USE_UNCERTAINTY = os.environ.get("USE_UNCERTAINTY", "false").lower() == "true"
IMG_SIZE = 256

_feature_maps = {}
_hooks = []


def register_hooks(net_G):
    # encoder groups are inside the ResNet body at layers.0.4 ... layers.0.7
    # decoder UnetBlocks are at layers.4 ... layers.7
    # bottleneck convolutions are at layers.3
    targets = {
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
    hooked = set()

    for full_name, module in net_G.named_modules():
        for suffix, label in targets.items():
            if full_name == suffix or full_name.endswith("." + suffix):
                def make_hook(key):
                    def hook(_mod, _inp, out):
                        _feature_maps[key] = out.detach().cpu()
                    return hook
                _hooks.append(module.register_forward_hook(make_hook(label)))
                hooked.add(label)
                break

    print(f"hooks registered on: {sorted(hooked)}")


def load_model(path, uncertainty):
    net_G = build_res_unet(n_input=1, n_output=2, size=IMG_SIZE)
    if uncertainty:
        net_G = UncertaintyGenerator(net_G)

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("generator_state_dict", "net_G", "model"):
            if key in state:
                state = state[key]
                break

    net_G.load_state_dict(state, strict=False)
    net_G.eval()
    register_hooks(net_G)
    return net_G


def array_to_b64(arr):
    arr = np.clip(arr, 0, 1)
    arr_u8 = (arr * 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_feature_channels(fmap_tensor, n=16, size=72):
    # normalize each channel to 0-1 independently and return as base64 list
    fmap = fmap_tensor[0]  # [C, H, W]
    total_channels = fmap.shape[0]
    n = min(n, total_channels)
    result = []
    for i in range(n):
        ch = fmap[i].numpy()
        mn, mx = ch.min(), ch.max()
        ch = (ch - mn) / (mx - mn + 1e-8)
        img = Image.fromarray((ch * 255).astype(np.uint8), "L").resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result, total_channels


def uncertainty_to_heatmap(std_norm):
    # blue = certain, red = uncertain
    r = std_norm
    g = 1 - np.abs(std_norm * 2 - 1)
    b = 1 - std_norm
    return np.stack([r, g, b], axis=-1)


@torch.no_grad()
def run_colorization(net_G, image_bytes, uncertainty):
    _feature_maps.clear()

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    rgb_np = np.array(pil_img).astype(np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb_np).astype(np.float32)

    L_raw  = lab[:, :, 0]
    L_norm = (L_raw / 50.0) - 1.0
    L_tensor = torch.tensor(L_norm)[None, None]

    if uncertainty:
        ab_pred_tensor, log_var_tensor = net_G(L_tensor)
    else:
        ab_pred_tensor = net_G(L_tensor)
        log_var_tensor = None

    ab_pred  = ab_pred_tensor[0].permute(1, 2, 0).numpy() * 110.0
    lab_pred = np.concatenate([L_raw[:, :, None], ab_pred], axis=-1)
    rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1)

    a_display = (ab_pred[:, :, 0] / 127.0 + 1) / 2
    b_display = (ab_pred[:, :, 1] / 127.0 + 1) / 2
    L_display = L_raw / 100.0

    uncertainty_heatmap_b64 = None
    uncertainty_overlay_b64 = None
    if log_var_tensor is not None:
        std = torch.exp(log_var_tensor / 2).mean(dim=1)[0].numpy()
        mn, mx = std.min(), std.max()
        std_norm = (std - mn) / (mx - mn + 1e-8)
        heatmap = uncertainty_to_heatmap(std_norm)
        uncertainty_heatmap_b64 = array_to_b64(heatmap.astype(np.float32))
        L_rgb = np.stack([L_display, L_display, L_display], axis=-1)
        overlay = np.clip(0.55 * L_rgb + 0.45 * heatmap, 0, 1)
        uncertainty_overlay_b64 = array_to_b64(overlay.astype(np.float32))

    # metadata for each layer — label shown in the diagram block
    fmap_meta = {
        "enc1":       {"label": "Encoder 1",   "desc": "Low-level edges and textures",                      "ch_hint": "64",  "sz_hint": "128×128"},
        "enc2":       {"label": "Encoder 2",   "desc": "Shapes and local structure",                         "ch_hint": "128", "sz_hint": "64×64"},
        "enc3":       {"label": "Encoder 3",   "desc": "Object parts and semantic regions",                  "ch_hint": "256", "sz_hint": "32×32"},
        "enc4":       {"label": "Encoder 4",   "desc": "High-level semantics — scene understanding",         "ch_hint": "512", "sz_hint": "16×16"},
        "bottleneck": {"label": "Bottleneck",  "desc": "Most compressed representation — full scene context","ch_hint": "512", "sz_hint": "8×8"},
        "dec4":       {"label": "Decoder 4",   "desc": "Begins recovering spatial detail from bottleneck",   "ch_hint": "512", "sz_hint": "16×16"},
        "dec3":       {"label": "Decoder 3",   "desc": "Recovering object-level color regions",              "ch_hint": "256", "sz_hint": "32×32"},
        "dec2":       {"label": "Decoder 2",   "desc": "Local color structure, guided by skip features",     "ch_hint": "128", "sz_hint": "64×64"},
        "dec1":       {"label": "Decoder 1",   "desc": "Full-resolution color prediction — texture detail",  "ch_hint": "64",  "sz_hint": "128×128"},
    }

    feature_maps_out = {}
    for key, meta in fmap_meta.items():
        if key in _feature_maps:
            tensor = _feature_maps[key]
            channels, total = extract_feature_channels(tensor, n=16, size=72)
            _, C, H, W = tensor.shape
            feature_maps_out[key] = {
                "channels":    channels,
                "total_ch":    total,
                "spatial":     f"{H}×{W}",
                "label":       meta["label"],
                "desc":        meta["desc"],
                "ch_hint":     meta["ch_hint"],
                "sz_hint":     meta["sz_hint"],
            }

    return {
        "input_L":             array_to_b64(L_display.astype(np.float32)),
        "original_rgb":        array_to_b64(rgb_np.astype(np.float32)),
        "colorized_rgb":       array_to_b64(rgb_pred.astype(np.float32)),
        "ab_a":                array_to_b64(a_display.astype(np.float32)),
        "ab_b":                array_to_b64(b_display.astype(np.float32)),
        "uncertainty_heatmap": uncertainty_heatmap_b64,
        "uncertainty_overlay": uncertainty_overlay_b64,
        "feature_maps":        feature_maps_out,
        "has_uncertainty":     uncertainty and log_var_tensor is not None,
    }


app = FastAPI()
net_G_global = None


@app.on_event("startup")
def startup():
    global net_G_global
    if os.path.exists(CHECKPOINT_PATH):
        print(f"loading model from {CHECKPOINT_PATH}")
        net_G_global = load_model(CHECKPOINT_PATH, USE_UNCERTAINTY)
        print("model ready")
    else:
        print(f"no checkpoint at {CHECKPOINT_PATH} — set CHECKPOINT_PATH env var")


@app.get("/api/health")
def health():
    return {"model_loaded": net_G_global is not None, "uncertainty": USE_UNCERTAINTY}


@app.post("/api/colorize")
async def colorize_endpoint(file: UploadFile = File(...)):
    if net_G_global is None:
        raise HTTPException(503, "model not loaded")
    data = await file.read()
    result = run_colorization(net_G_global, data, USE_UNCERTAINTY)
    return JSONResponse(result)


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
