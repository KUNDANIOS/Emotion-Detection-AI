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
MODEL_PATH = "emotion_vit_model.pt"
# Hugging Face model URL
HF_MODEL_URL = "https://huggingface.co/Kundannn/emotion-vit-model/resolve/main/emotion_vit_model.pt"
EMOTIONS   = ["angry", "happy", "sad"]
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
# ————————————————————————————————————————————————————————————————

# Download model from Hugging Face
def download_model_from_huggingface(url, destination):
    print(f"Downloading model from Hugging Face...")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"Total size: {total_size / (1024*1024):.1f} MB")
        
        downloaded = 0
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress every 50MB
                    if downloaded % (50 * 1024 * 1024) == 0:
                        print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
        
        file_size = os.path.getsize(destination)
        print(f"✓ Model downloaded successfully! Size: {file_size / (1024*1024):.1f} MB")
        
        # Verify file size
        if file_size < 50 * 1024 * 1024:
            raise Exception(f"Downloaded file too small ({file_size / (1024*1024):.1f} MB). Expected ~344 MB")
        
    except Exception as e:
        print(f"ERROR downloading model: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

# Download model if it doesn't exist
model = None
if not os.path.exists(MODEL_PATH):
    try:
        print("Model file not found locally. Attempting to download...")
        download_model_from_huggingface(HF_MODEL_URL, MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Could not download model: {e}")
        print("The app will start but emotion detection will not work.")
        print("Please check the Hugging Face URL and try again.")
else:
    print(f"Model already exists at {MODEL_PATH}")

# Device & face detector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

mtcnn  = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.5, 0.6, 0.7], min_face_size=40)

# Load model with timeout protection
print("=" * 60)
print("STARTING MODEL LOADING PROCESS")
print("=" * 60)
print(f"Looking for model at: {MODEL_PATH}")
print(f"Model exists locally: {os.path.exists(MODEL_PATH)}")
print(f"PyTorch version: {torch.__version__}")

model = None

try:
    if not os.path.exists(MODEL_PATH):
        print("Model file not found locally. Attempting to download...")
        download_model_from_huggingface(HF_MODEL_URL, MODEL_PATH)
    else:
        print(f"Model already exists at {MODEL_PATH}")
        file_size = os.path.getsize(MODEL_PATH) / (1024*1024)
        print(f"Model file size: {file_size:.1f} MB")
    
    print("Loading emotion detection model (this may take 30-60 seconds)...")
    import time
    start_time = time.time()
    
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.to(DEVICE).eval()
    
    load_time = time.time() - start_time
    print(f"✓ Model loaded successfully in {load_time:.1f} seconds!")
    print("=" * 60)
    
except RuntimeError as e:
    import traceback
    print("=" * 60)
    print("❌ PYTORCH RUNTIME ERROR:")
    print(f"Error: {e}")
    if "scaled_dot_product_attention" in str(e):
        print("\n⚠️  MODEL INCOMPATIBILITY DETECTED!")
        print(f"Current PyTorch version: {torch.__version__}")
        print("Required: PyTorch 2.5+")
    print("Full traceback:")
    print(traceback.format_exc())
    print("=" * 60)
    model = None
    
except Exception as e:
    import traceback
    print("=" * 60)
    print("❌ ERROR LOADING MODEL:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
    print("Full traceback:")
    print(traceback.format_exc())
    print("=" * 60)
    model = None

if model is None:
    print("\n⚠️  WARNING: Model not loaded. App will run but detection won't work.")
    print("Check the errors above for details.")
else:
    print("\n✓ Model ready for inference!")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def process_pil(img: Image.Image):
    if model is None:
        raise RuntimeError("Model not loaded. Check server logs.")
    
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

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }), 200

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)