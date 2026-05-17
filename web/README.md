# Colorization Explorer — web interface

Interactive visualization of the GAN colorization pipeline.
Upload a grayscale (or color) image and the platform shows:
- LAB decomposition: L input, predicted a and b channels, final RGB
- Encoder feature maps at each of the 4 ResNet-18 groups (click any to enlarge)
- Uncertainty heatmap showing where the model is confident vs unsure

## Which checkpoint to use

There are two model variants, both saved in Google Drive:

**Option A — base model (no uncertainty)**
- Drive path: `colorization_checkpoints/pretrained_generator.pth`
- This is the GAN-trained generator without the uncertainty head
- Faster to load, no uncertainty heatmap

**Option B — uncertainty model (recommended)**
- Drive path: `colorization_checkpoints_uncertainty/checkpoint_epoch_N.pth`
- Use the highest epoch number available (e.g. `checkpoint_epoch_20.pth`)
- This model also predicts per-pixel confidence — enables the uncertainty heatmap
- Needs `USE_UNCERTAINTY=true` when starting the server

The app auto-detects the checkpoint format so you don't need to change any code.

## Running locally

1. Download the checkpoint from Drive and save it as `web/model.pt` in the project folder
   (or keep it anywhere and point to it with `CHECKPOINT_PATH`)

2. Install dependencies (from the project root):
   ```
   pip install -r web/requirements.txt
   ```

3. Start the server — from the project root:

   For the base model:
   ```
   uvicorn web.app:app --reload --port 8000
   ```

   For the uncertainty model:
   ```
   set USE_UNCERTAINTY=true
   uvicorn web.app:app --reload --port 8000
   ```

4. Open `http://localhost:8000`

## Streamlit interface

A Streamlit demo is also available in `web/streamlit_app.py`. It reuses the same generator logic and provides a quick interactive interface for your thesis presentation.

Run locally from the project root:
```bash
pip install -r web/requirements.txt
streamlit run web/streamlit_app.py
```

If you want to load a custom checkpoint from a different path:
```bash
set CHECKPOINT_PATH=C:\path\to\checkpoint.pth
streamlit run web/streamlit_app.py
```

The app will automatically detect uncertainty checkpoints from the filename.

## Deploying to HuggingFace Spaces (free hosting)

1. Create a new Space on huggingface.co — choose type **Docker**
2. Upload all project files
3. Upload `model.pt` (HuggingFace uses Git LFS for large files automatically)
4. Add a `Dockerfile` at the project root:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r web/requirements.txt
ENV CHECKPOINT_PATH=/app/web/model.pt
ENV USE_UNCERTAINTY=true
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

HuggingFace Spaces expose port 7860 by default.
CPU inference takes roughly 3-8 seconds per image — acceptable for a demo.

## Custom domain

Buy a `.com` or `.ro` domain (~12 euros/year on Namecheap or GoDaddy) and point it
to the HuggingFace Space URL using a CNAME DNS record. That's the only cost.
