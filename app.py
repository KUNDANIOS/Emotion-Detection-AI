import io
import os
import tempfile
import requests
import base64
import numpy as np
import cv2

from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

# —— CONFIGURATION —————————————————————————————————————————————
MODEL_PATH = r"C:/Users/kunda/OneDrive/Desktop/New folder/ML_sem/emotion_vit_model.pt"
EMOTIONS   = ["angry", "happy", "sad"]
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
# ————————————————————————————————————————————————————————————————

# Device & face detector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn  = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.5, 0.6, 0.7], min_face_size=40)

# Load model
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE).eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def process_pil(img: Image.Image):
    boxes, probs = mtcnn.detect(img)
    results = []

    if boxes is None:
        return results

    for box in boxes:
        face = img.crop(box)
        tensor = preprocess(face).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            idx = torch.argmax(logits, dim=1).item()
            label = EMOTIONS[idx]

        results.append({
            "box": [float(b) for b in box],
            "expression": label
        })

    return results

def process_camera_frame(frame: np.ndarray):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    return process_pil(pil_img)

# ─── Flask setup ────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files['file']
    try:
        img = Image.open(f.stream).convert("RGB")
        expressions = process_pil(img)
        return jsonify({"expressions": expressions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json(force=True)
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url']
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        expressions = process_pil(img)
        return jsonify({"expressions": expressions}), 200
    except Exception as e:
        return jsonify({"error": f"Could not fetch image: {e}"}), 400

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = request.get_json(force=True)
    if not data or 'image_base64' not in data:
        return jsonify({"error": "Missing image_base64"}), 400

    try:
        image_data = base64.b64decode(data['image_base64'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Image decoding failed")
        expressions = process_camera_frame(frame)
        return jsonify({"expressions": expressions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # <--- important to set host=0.0.0.0
