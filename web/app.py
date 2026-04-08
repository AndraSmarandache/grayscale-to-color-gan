import os
import sys
import io
import base64
import numpy as np
import torch
from PIL import Image
from skimage import color as skcolor
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.generator import build_res_unet, UncertaintyGenerator

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
IMG_SIZE   = 256

_feature_maps = {}
_hooks        = []
net_G_global  = None
current_path  = None
current_unc   = False


def is_uncertainty_model(filename):
    return "uncertainty" in os.path.basename(filename).lower()


def get_available_models():
    seen   = set()
    result = []
    for directory in [MODELS_DIR, os.path.join(ROOT, "web")]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not (fname.endswith(".pth") or fname.endswith(".pt")):
                continue
            fpath = os.path.normpath(os.path.join(directory, fname))
            if fpath in seen:
                continue
            seen.add(fpath)
            result.append({
                "name":        fname,
                "path":        fpath,
                "uncertainty": is_uncertainty_model(fname),
            })
    return result


def _default_checkpoint():
    env = os.environ.get("CHECKPOINT_PATH", "")
    if env and os.path.exists(env):
        return env
    models = get_available_models()
    for m in models:
        if "pretrained" in m["name"].lower():
            return m["path"]
    return models[0]["path"] if models else ""


def register_hooks(net_G):
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
    print(f"hooks registered: {sorted(hooked)}")


def _build_generator(path, uncertainty):
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
    return net_G


def load_model(path, uncertainty):
    net_G = _build_generator(path, uncertainty)
    register_hooks(net_G)
    return net_G


def array_to_b64(arr):
    arr    = np.clip(arr, 0, 1)
    arr_u8 = (arr * 255).astype(np.uint8)
    mode   = "L" if arr.ndim == 2 else "RGB"
    buf    = io.BytesIO()
    Image.fromarray(arr_u8, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_feature_channels(fmap_tensor, n=9999, size=48):
    fmap           = fmap_tensor[0]
    total_channels = fmap.shape[0]
    n              = min(n, total_channels)
    result         = []
    for i in range(n):
        ch     = fmap[i].numpy()
        mn, mx = ch.min(), ch.max()
        ch     = (ch - mn) / (mx - mn + 1e-8)
        img    = Image.fromarray((ch * 255).astype(np.uint8), "L").resize(
                     (size, size), Image.BILINEAR)
        buf    = io.BytesIO()
        img.save(buf, format="PNG")
        result.append(base64.b64encode(buf.getvalue()).decode())
    return result, total_channels


def uncertainty_to_heatmap(std_norm):
    r = std_norm
    g = 1 - np.abs(std_norm * 2 - 1)
    b = 1 - std_norm
    return np.stack([r, g, b], axis=-1)


def _prepare_L(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
                  (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    rgb_np  = np.array(pil_img).astype(np.float64) / 255.0
    lab     = skcolor.rgb2lab(rgb_np).astype(np.float32)
    L_raw   = lab[:, :, 0]
    L_norm  = (L_raw / 50.0) - 1.0
    return rgb_np, L_raw, torch.tensor(L_norm)[None, None]


@torch.no_grad()
def run_colorization(net_G, image_bytes, uncertainty):
    _feature_maps.clear()

    rgb_np, L_raw, L_tensor = _prepare_L(image_bytes)

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
        std      = torch.exp(log_var_tensor / 2).mean(dim=1)[0].numpy()
        mn, mx   = std.min(), std.max()
        std_norm = (std - mn) / (mx - mn + 1e-8)
        heatmap  = uncertainty_to_heatmap(std_norm)
        uncertainty_heatmap_b64 = array_to_b64(heatmap.astype(np.float32))
        L_rgb    = np.stack([L_display] * 3, axis=-1)
        overlay  = np.clip(0.55 * L_rgb + 0.45 * heatmap, 0, 1)
        uncertainty_overlay_b64 = array_to_b64(overlay.astype(np.float32))

    fmap_meta = {
        "enc1":       {"label": "Encoder 1",  "desc": "Low-level edges and textures"},
        "enc2":       {"label": "Encoder 2",  "desc": "Shapes and local structure"},
        "enc3":       {"label": "Encoder 3",  "desc": "Object parts and semantic regions"},
        "enc4":       {"label": "Encoder 4",  "desc": "High-level semantics"},
        "bottleneck": {"label": "Bottleneck", "desc": "Most compressed representation"},
        "dec4":       {"label": "Decoder 4",  "desc": "Begins recovering spatial detail"},
        "dec3":       {"label": "Decoder 3",  "desc": "Recovering object-level color regions"},
        "dec2":       {"label": "Decoder 2",  "desc": "Local color via skip features"},
        "dec1":       {"label": "Decoder 1",  "desc": "Full-resolution color prediction"},
    }

    feature_maps_out = {}
    for key, meta in fmap_meta.items():
        if key not in _feature_maps:
            continue
        tensor   = _feature_maps[key]
        channels, total = extract_feature_channels(tensor)
        _, C, H, W      = tensor.shape
        feature_maps_out[key] = {
            "channels": channels,
            "total_ch": total,
            "spatial":  f"{H}x{W}",
            "label":    meta["label"],
            "desc":     meta["desc"],
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


@app.on_event("startup")
def startup():
    global net_G_global, current_path, current_unc
    path = _default_checkpoint()
    if path and os.path.exists(path):
        print(f"loading model: {path}")
        current_unc  = is_uncertainty_model(path)
        net_G_global = load_model(path, current_unc)
        current_path = path
        print("model ready")
    else:
        print("no model found — set CHECKPOINT_PATH or place .pth files in models/")


@app.get("/api/health")
def health():
    return {
        "model_loaded":  net_G_global is not None,
        "current_model": os.path.basename(current_path) if current_path else None,
        "uncertainty":   current_unc,
    }


@app.get("/api/models")
def list_models():
    return {
        "models":       get_available_models(),
        "current_path": current_path,
        "uncertainty":  current_unc,
    }


@app.post("/api/load_model")
async def load_model_endpoint(req: Request):
    global net_G_global, current_path, current_unc
    body = await req.json()
    path = body.get("path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(400, "model file not found")
    try:
        for h in _hooks:
            h.remove()
        _hooks.clear()
        _feature_maps.clear()
        use_unc      = bool(body.get("uncertainty", is_uncertainty_model(path)))
        net_G_global = load_model(path, use_unc)
        current_path = path
        current_unc  = use_unc
        return {"ok": True, "name": os.path.basename(path), "uncertainty": use_unc}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/compare")
async def compare_endpoint(file: UploadFile = File(...)):
    """Run inference with every available model and return colorized results side by side."""
    data    = await file.read()
    models  = get_available_models()
    results = []

    _, L_raw, L_tensor = _prepare_L(data)
    rgb_np, _, _       = _prepare_L(data)

    for m in models:
        try:
            net = _build_generator(m["path"], m["uncertainty"])
            with torch.no_grad():
                if m["uncertainty"]:
                    ab_pred_tensor, _ = net(L_tensor)
                else:
                    ab_pred_tensor = net(L_tensor)
            ab_pred  = ab_pred_tensor[0].permute(1, 2, 0).numpy() * 110.0
            lab_pred = np.concatenate([L_raw[:, :, None], ab_pred], axis=-1)
            rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1)
            results.append({
                "name":          m["name"],
                "uncertainty":   m["uncertainty"],
                "colorized_rgb": array_to_b64(rgb_pred.astype(np.float32)),
                "error":         None,
            })
        except Exception as e:
            results.append({
                "name":          m["name"],
                "uncertainty":   m["uncertainty"],
                "colorized_rgb": None,
                "error":         str(e),
            })

    original = array_to_b64(rgb_np.astype(np.float32))
    return JSONResponse({"original_rgb": original, "results": results})


@app.post("/api/colorize")
async def colorize_endpoint(file: UploadFile = File(...)):
    if net_G_global is None:
        raise HTTPException(503, "model not loaded")
    data   = await file.read()
    result = run_colorization(net_G_global, data, current_unc)
    return JSONResponse(result)


app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static",
)
