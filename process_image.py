import torch
from torchvision import transforms
from PIL import Image

# 1. Config
model_path = "emotion_vit_model.pt"   # TorchScript model
image_path = "C:/Users/itssh/Downloads/WhatsApp Image 2025-04-22 at 20.08.29.jpeg"  # Test image path
class_labels = ["angry", "happy", "sad"]

# 2. Preprocessing (same as ViTImageProcessor)
preprocess = transforms.Compose([
    transforms.Resize(256),                
    transforms.CenterCrop(224),           
    transforms.ToTensor(),                
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# 3. Load image & preprocess
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, 3, 224, 224]

# 4. Load TorchScript model
model = torch.jit.load(model_path, map_location="cpu")
model.eval()

# 5. Inference
with torch.no_grad():
    logits = model(input_tensor)  # model output
    print(f"Raw model output (logits): {logits}")
    print(f"Logits shape: {logits.shape}")

    pred_idx = logits.argmax(dim=-1).item()

# 6. Output
if 0 <= pred_idx < len(class_labels):
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted label: {class_labels[pred_idx]}")
else:
    print(f"Error: predicted index {pred_idx} is out of range for class_labels!")
