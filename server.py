import base64, io, os, time
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-so400m-patch14-384"

model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
app = FastAPI()


def load_image(src):
    if src.startswith("http"):
        data = requests.get(src, timeout=15).content
    else:
        data = base64.b64decode(src.split(",")[-1])
    return Image.open(io.BytesIO(data)).convert("RGB")


def features(out):
    return out if torch.is_tensor(out) else out.pooler_output


def embed_images(images):
    with torch.inference_mode():
        inputs = processor.image_processor(images, return_tensors="pt", device="cuda")
        pixels = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
        return features(model.get_image_features(pixel_values=pixels)).float().cpu().numpy()


def embed_texts(texts):
    with torch.inference_mode():
        inputs = processor(
            text=[t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to("cuda")
        return features(model.get_text_features(**inputs)).float().cpu().numpy()


embed_images([Image.new("RGB", (384, 384))])
embed_texts(["warmup"])


@app.get("/ping")
def ping():
    return {"status": "healthy"}


@app.post("/embed")
def embed(body: dict):
    started = time.perf_counter()
    if body.get("images"):
        images = [load_image(s) for s in body["images"]]
        decoded = time.perf_counter()
        vecs = embed_images(images)
    else:
        decoded = time.perf_counter()
        vecs = embed_texts(body["texts"])
    encoded = time.perf_counter()
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return {
        "embeddings": vecs.tolist(),
        "timing": {
            "decode_ms": (decoded - started) * 1000,
            "embed_ms": (encoded - decoded) * 1000,
            "total_ms": (time.perf_counter() - started) * 1000,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
