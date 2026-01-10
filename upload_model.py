from huggingface_hub import HfApi, create_repo, login
import os

# First, login with your token
print("Please enter your Hugging Face token:")
print("(Get it from: https://huggingface.co/settings/tokens)")
token = input("Token: ").strip()

# Login
login(token=token)
print("✓ Logged in successfully!")

# Configuration
MODEL_PATH = "emotion_vit_model.pt"
# Replace 'your-username' with your actual Hugging Face username
username = input("\nEnter your Hugging Face username: ").strip()
REPO_NAME = f"{username}/emotion-vit-model"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file '{MODEL_PATH}' not found!")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    exit(1)

file_size = os.path.getsize(MODEL_PATH)
print(f"\n📦 Model file found! Size: {file_size / (1024*1024):.1f} MB")

# Initialize API
api = HfApi()

# Create repository
print(f"\n🔨 Creating repository: {REPO_NAME}")
try:
    create_repo(REPO_NAME, repo_type="model", private=False, exist_ok=True)
    print(f"✓ Repository ready!")
except Exception as e:
    print(f"Note: {e}")

# Upload the model file
print("\n📤 Uploading model file (this may take 5-10 minutes for 344MB)...")
try:
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="emotion_vit_model.pt",
        repo_id=REPO_NAME,
        repo_type="model",
    )
    print(f"\n✅ SUCCESS! Model uploaded!")
    print(f"\n📍 Your model URL:")
    print(f"   https://huggingface.co/{REPO_NAME}")
    print(f"\n📍 Direct download URL (use this in app.py):")
    print(f"   https://huggingface.co/{REPO_NAME}/resolve/main/emotion_vit_model.pt")
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    exit(1)