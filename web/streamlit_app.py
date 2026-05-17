import io
import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage import color as skcolor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.generator import UncertaintyGenerator, build_res_unet

IMG_SIZE = 256
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))


def is_uncertainty_model(filename: str) -> bool:
    return "uncertainty" in os.path.basename(filename).lower()


@st.cache_data(show_spinner=False)
def get_available_models() -> list[dict]:
    seen = set()
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
                "name": os.path.basename(fname),
                "path": fpath,
                "uncertainty": is_uncertainty_model(fname),
            })
    return result


@st.cache_resource(show_spinner=False)
def load_model(path: str, uncertainty: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = build_res_unet(n_input=1, n_output=2, size=IMG_SIZE)
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


def _prepare_L(image_bytes: bytes):
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.LANCZOS
    )
    rgb_np = np.array(pil_img).astype(np.float64) / 255.0
    lab = skcolor.rgb2lab(rgb_np).astype(np.float32)
    L_raw = lab[:, :, 0]
    L_norm = (L_raw / 50.0) - 1.0
    return rgb_np, L_raw, torch.tensor(L_norm)[None, None]


def array_to_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 1)
    arr_u8 = (arr * 255).astype(np.uint8)
    mode = "L" if arr.ndim == 2 else "RGB"
    return Image.fromarray(arr_u8, mode=mode)


def extract_feature_channels(fmap_tensor: torch.Tensor, n: int = 12, size: int = 64):
    fmap = fmap_tensor[0].cpu().detach().numpy()
    total = fmap.shape[0]
    channels = min(n, total)
    images = []

    for i in range(channels):
        ch = fmap[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        img = Image.fromarray((ch * 255).astype(np.uint8), "L").resize((size, size), Image.BILINEAR)
        images.append(img)

    return images, total


def uncertainty_to_heatmap(std_norm: np.ndarray) -> np.ndarray:
    r = std_norm
    g = 1 - np.abs(std_norm * 2 - 1)
    b = 1 - std_norm
    return np.stack([r, g, b], axis=-1)


@torch.no_grad()
def run_colorization(net_G, image_bytes: bytes, uncertainty: bool) -> dict:
    rgb_np, L_raw, L_tensor = _prepare_L(image_bytes)

    if uncertainty:
        ab_pred_tensor, log_var_tensor = net_G(L_tensor)
    else:
        ab_pred_tensor = net_G(L_tensor)
        log_var_tensor = None

    ab_pred = ab_pred_tensor[0].permute(1, 2, 0).cpu().numpy() * 110.0
    lab_pred = np.concatenate([L_raw[:, :, None], ab_pred], axis=-1)
    rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1)

    a_display = (ab_pred[:, :, 0] / 127.0 + 1) / 2
    b_display = (ab_pred[:, :, 1] / 127.0 + 1) / 2
    L_display = L_raw / 100.0

    uncertainty_heatmap = None
    uncertainty_overlay = None
    if log_var_tensor is not None:
        std = torch.exp(log_var_tensor / 2).mean(dim=1)[0].cpu().numpy()
        mn, mx = std.min(), std.max()
        std_norm = (std - mn) / (mx - mn + 1e-8)
        uncertainty_heatmap = array_to_image(uncertainty_to_heatmap(std_norm).astype(np.float32))
        L_rgb = np.stack([L_display] * 3, axis=-1)
        overlay = np.clip(0.55 * L_rgb + 0.45 * uncertainty_to_heatmap(std_norm), 0, 1)
        uncertainty_overlay = array_to_image(overlay.astype(np.float32))

    return {
        "input_L": array_to_image(L_display.astype(np.float32)),
        "original_rgb": array_to_image(rgb_np.astype(np.float32)),
        "colorized_rgb": array_to_image(rgb_pred.astype(np.float32)),
        "ab_a": array_to_image(a_display.astype(np.float32)),
        "ab_b": array_to_image(b_display.astype(np.float32)),
        "uncertainty_heatmap": uncertainty_heatmap,
        "uncertainty_overlay": uncertainty_overlay,
        "has_uncertainty": uncertainty and log_var_tensor is not None,
    }


def compare_models(image_bytes: bytes, models: list[dict]) -> dict:
    _, L_raw, L_tensor = _prepare_L(image_bytes)
    rgb_np = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    rgb_np = rgb_np.astype(np.float32) / 255.0

    results = []
    for model_info in models:
        try:
            net = load_model(model_info["path"], model_info["uncertainty"])
            with torch.no_grad():
                if model_info["uncertainty"]:
                    ab_pred_tensor, _ = net(L_tensor)
                else:
                    ab_pred_tensor = net(L_tensor)
            ab_pred = ab_pred_tensor[0].permute(1, 2, 0).cpu().numpy() * 110.0
            lab_pred = np.concatenate([L_raw[:, :, None], ab_pred], axis=-1)
            rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1)
            results.append(
                {
                    "name": model_info["name"],
                    "uncertainty": model_info["uncertainty"],
                    "colorized": array_to_image(rgb_pred.astype(np.float32)),
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "name": model_info["name"],
                    "uncertainty": model_info["uncertainty"],
                    "colorized": None,
                    "error": str(exc),
                }
            )

    return {"original": array_to_image(rgb_np), "results": results}


def main() -> None:
    st.set_page_config(
        page_title="Image Colorization",
        page_icon="🎨",
        layout="wide",
    )

    st.title("Image Colorization Explorer")
    st.write(
        "Upload a grayscale or color image and the app will predict LAB color channels, reconstruct the color image, "
        "and optionally show uncertainty outputs for uncertainty-aware checkpoints."
    )

    models = get_available_models()
    if not models:
        st.warning(
            "No model checkpoints found. Place `.pth` or `.pt` files in `models/` or set `CHECKPOINT_PATH` before running this app."
        )
        return

    model_names = [
        f"{m['name']} {'(uncertainty)' if m['uncertainty'] else '(base)'}" for m in models
    ]
    selected_index = st.sidebar.selectbox("Choose a checkpoint", list(range(len(models))), format_func=lambda i: model_names[i])
    selected_model = models[selected_index]

    st.sidebar.markdown("---")
    st.sidebar.write("**Selected model**")
    st.sidebar.info(selected_model["name"])
    if selected_model["uncertainty"]:
        st.sidebar.success("Uncertainty head enabled")
    else:
        st.sidebar.info("Base generator only")

    st.sidebar.markdown("---")
    st.sidebar.write(
        "If you want to use a checkpoint outside the default folders, set `CHECKPOINT_PATH` and restart the app."
    )

    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    if checkpoint_path:
        st.sidebar.write(f"Current CHECKPOINT_PATH: `{checkpoint_path}`")

    try:
        model = load_model(selected_model["path"], selected_model["uncertainty"])
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    if not uploaded_file:
        st.info("Upload a photo to begin. The image will be resized to 256×256 for the generator.")
        return

    image_bytes = uploaded_file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.markdown("### Original input")
    st.image(original_image, use_column_width=True)

    if st.button("Colorize image"):
        with st.spinner("Running inference..."):
            result = run_colorization(model, image_bytes, selected_model["uncertainty"])

        st.markdown("## Colorization results")
        cols = st.columns(4)
        cols[0].image(result["input_L"], caption="L channel")
        cols[1].image(result["ab_a"], caption="Predicted a channel")
        cols[2].image(result["ab_b"], caption="Predicted b channel")
        cols[3].image(result["colorized_rgb"], caption="Final colorization")

        st.markdown("### Full colorization")
        st.image(result["colorized_rgb"], use_column_width=True)

        if result["has_uncertainty"]:
            st.markdown("### Uncertainty visualization")
            ucols = st.columns(2)
            ucols[0].image(result["uncertainty_heatmap"], caption="Uncertainty heatmap")
            ucols[1].image(result["uncertainty_overlay"], caption="Uncertainty overlay")
            st.caption("Brighter values indicate higher model uncertainty.")

        with st.expander("Model comparison across all available checkpoints"):
            if st.button("Compare all checkpoints"):
                with st.spinner("Comparing models..."):
                    compare = compare_models(image_bytes, models)
                st.markdown("#### Reference image")
                st.image(compare["original"], use_column_width=True)
                grid_cols = st.columns(min(len(models), 3))
                for idx, row in enumerate(compare["results"]):
                    col = grid_cols[idx % len(grid_cols)]
                    if row["error"]:
                        col.error(f"{row['name']}: {row['error']}")
                    else:
                        col.image(row["colorized"], caption=row["name"])

    st.markdown("---")
    st.write(
        "This Streamlit app uses the same colorization generator logic as the existing FastAPI interface, "
        "but offers a faster interactive demo experience for your thesis presentation."
    )


if __name__ == "__main__":
    main()
