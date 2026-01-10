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
GDRIVE_FILE_ID = "1hi_Q56qsuOk5Ke_OkvVKrgG8JlMwPexJ"
EMOTIONS   = ["angry", "happy", "sad"]
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
# ————————————————————————————————————————————————————————————————

# Download model from Google Drive using requests with retry logic
def download_model_from_gdrive(file_id, destination):
    print(f"Downloading model from Google Drive...")
    
    # Try multiple URL formats
    urls = [
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={file_id}",
    ]
    
    for url in urls:
        try:
            print(f"Trying URL: {url[:50]}...")
            session = requests.Session()
            
            # First request might return a confirmation page for large files
            response = session.get(url, stream=True, timeout=60)
            
            # Check for virus scan warning page
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = url + f"&confirm={value}"
                    response = session.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                # Save the file
                total_size = 0
                print("Starting download...")
                with open(destination, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                            if total_size % (50 * 1024 * 1024) == 0:  # Print every 50MB
                                print(f"Downloaded {total_size / (1024*1024):.1f} MB...")
                
                file_size = os.path.getsize(destination)
                print(f"Model downloaded successfully! Size: {file_size / (1024*1024):.1f} MB")
                
                # Verify the file is not too small (should be ~344MB)
                if file_size < 100 * 1024 * 1024:  # Less than 100MB
                    print(f"Warning: File seems too small ({file_size / (1024*1024):.1f} MB). Expected ~344 MB")
                    os.remove(destination)
                    continue
                
                return
            else:
                print(f"Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Error with this URL: {e}")
            if os.path.exists(destination):
                os.remove(destination)
            continue
    
    raise Exception("Failed to download model from all attempted URLs")

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    try:
        download_model_from_gdrive(GDRIVE_FILE_ID, MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Could not download model: {e}")
        print("Please ensure the model file is accessible or use environment variables")
else:
    print(f"Model already exists at {MODEL_PATH}")

# Device & face detector
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

mtcnn  = MTCNN(keep_all=True, device=DEVICE, thresholds=[0.5, 0.6, 0.7], min_face_size=40)

# Load model
print("Loading emotion detection model...")
try:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.to(DEVICE).eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

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