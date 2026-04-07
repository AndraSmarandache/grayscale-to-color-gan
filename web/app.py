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

# add project root to path so we can import from src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.generator import build_res_unet, UncertaintyGenerator

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join(ROOT, "web", "model.pt"))
USE_UNCERTAINTY = os.environ.get("USE_UNCERTAINTY", "false").lower() == "true"
IMG_SIZE = 256

# forward hooks store intermediate encoder outputs here
_feature_maps = {}
_hooks = []


def register_hooks(net_G):
    # fastai's DynamicUnet stores the ResNet body as layers.0
    # the 4 ResNet groups are at indices 4,5,6,7 inside that Sequential
    # so full names are layers.0.4 ... layers.0.7
    # if wrapped in UncertaintyGenerator they become base.layers.0.4 etc.
    targets = {
        "layers.0.4": "layer1",
        "layers.0.5": "layer2",
        "layers.0.6": "layer3",
        "layers.0.7": "layer4",
    }
    hooked = set()

    for full_name, module in net_G.named_modules():
        # match either direct or wrapped (base.layers.0.4)
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

    # checkpoints are saved with different key names depending on when/how they were saved
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
    # arr is float32 numpy, values 0-1, shape [H,W] or [H,W,3]
    arr = np.clip(arr, 0, 1)
    arr_u8 = (arr * 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr_u8, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_feature_grid(fmap_tensor, n=8, size=80):
    # take first n channels from [1, C, H, W], normalize each to 0-1, return as base64 list
    fmap = fmap_tensor[0]  # [C, H, W]
    n = min(n, fmap.shape[0])
    result = []
    for i in range(n):
        ch = fmap[i].numpy()
        mn, mx = ch.min(), ch.max()
        ch = (ch - mn) / (mx - mn + 1e-8)
        img = Image.fromarray((ch * 255).astype(np.uint8), "L").resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result


def uncertainty_to_heatmap(std_norm):
    # blue = certain, red = uncertain, green = middle
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

    L_raw = lab[:, :, 0]                       # luminance, range 0-100
    L_norm = (L_raw / 50.0) - 1.0              # normalize to -1..1 like during training
    L_tensor = torch.tensor(L_norm)[None, None]  # [1, 1, H, W]

    if uncertainty:
        ab_pred_tensor, log_var_tensor = net_G(L_tensor)
    else:
        ab_pred_tensor = net_G(L_tensor)
        log_var_tensor = None

    # convert predicted ab back to RGB
    ab_pred = ab_pred_tensor[0].permute(1, 2, 0).numpy() * 110.0  # denormalize
    lab_pred = np.concatenate([L_raw[:, :, None], ab_pred], axis=-1)
    rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1)

    # normalize ab channels to 0-1 just for display purposes
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
        # blend uncertainty heatmap over the grayscale input
        L_rgb = np.stack([L_display, L_display, L_display], axis=-1)
        overlay = np.clip(0.55 * L_rgb + 0.45 * heatmap, 0, 1)
        uncertainty_overlay_b64 = array_to_b64(overlay.astype(np.float32))

    # feature maps for each encoder level with a short description for the UI
    fmap_meta = {
        "layer1": {"label": "Group 1  —  64 channels  128×128", "desc": "Low-level edges and textures"},
        "layer2": {"label": "Group 2  —  128 channels  64×64",  "desc": "Shapes and local structure"},
        "layer3": {"label": "Group 3  —  256 channels  32×32",  "desc": "Object parts and semantic regions"},
        "layer4": {"label": "Group 4  —  512 channels  16×16",  "desc": "High-level semantics — what the network 'knows' about the scene"},
    }
    feature_maps_out = {}
    for key, meta in fmap_meta.items():
        if key in _feature_maps:
            feature_maps_out[key] = {
                "channels": extract_feature_grid(_feature_maps[key], n=8, size=80),
                "label": meta["label"],
                "desc": meta["desc"],
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
