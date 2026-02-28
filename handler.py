import os
import base64
import subprocess
from io import BytesIO

import runpod
from PIL import Image

WEIGHTS_DIR = "/app/weights"


def download_weights():
    """Download weights if not already present."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    weight_file = os.path.join(WEIGHTS_DIR, "model.safetensors")
    if not os.path.exists(weight_file):
        print("Downloading model weights...")
        subprocess.run(
            ["python3", "scripts/download_weights.py", "--weights-dir", WEIGHTS_DIR],
            check=True,
            cwd="/app"
        )
        print("Weights downloaded successfully.")
    else:
        print("Weights already present, skipping download.")


def load_pipeline():
    from fashn_vton import TryOnPipeline
    print("Loading TryOnPipeline...")
    pipeline = TryOnPipeline(weights_dir=WEIGHTS_DIR)
    print("Pipeline loaded.")
    return pipeline


def base64_to_image(b64_str: str) -> Image.Image:
    # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Initialize at cold start ──────────────────────────────────────────────────
download_weights()
PIPELINE = load_pipeline()


# ── RunPod handler ────────────────────────────────────────────────────────────
def handler(job):
    job_input = job.get("input", {})

    # Validate required fields
    if "person_image" not in job_input:
        return {"error": "Missing required field: person_image"}
    if "garment_image" not in job_input:
        return {"error": "Missing required field: garment_image"}

    category = job_input.get("category", "tops")
    if category not in ("tops", "bottoms", "one-pieces"):
        return {"error": f"Invalid category '{category}'. Must be: tops | bottoms | one-pieces"}

    try:
        person = base64_to_image(job_input["person_image"])
        garment = base64_to_image(job_input["garment_image"])
    except Exception as e:
        return {"error": f"Failed to decode images: {str(e)}"}

    try:
        result = PIPELINE(
            person_image=person,
            garment_image=garment,
            category=category,
        )
        output_b64 = image_to_base64(result.images[0])
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    return {"output_image": output_b64}


runpod.serverless.start({"handler": handler})