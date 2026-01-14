 😊 Emotion Detection AI

An advanced emotion detection system powered by deep learning that analyzes facial expressions in real-time.

## 🎯 Features

- **Real-time Detection**: Instant emotion analysis from images
- **3 Emotions**: Angry 😠, Happy 😊, Sad 😢
- **Multiple Input Methods**: Upload, webcam, or sample images
- **High Accuracy**: Based on Vision Transformer architecture
- **Easy to Use**: Simple, intuitive interface

## 🚀 How to Use

1. **Upload an Image**: Click the upload area or drag & drop
2. **Use Webcam**: Click webcam icon for live capture
3. **Try Samples**: Click sample buttons for demo images
4. **Detect**: Click "Detect Emotion" button
5. **View Results**: See detected emotion with confidence scores

## 🤖 Model Details

- **Architecture**: Vision Transformer (ViT)
- **Face Detection**: MTCNN
- **Framework**: PyTorch
- **Model Size**: ~344MB
- **Inference Time**: 1-2 seconds
<img width="1904" height="925" alt="image" src="https://github.com/user-attachments/assets/281b9bd8-acac-4be3-b0a3-556a0ef77997" />

<img width="1903" height="887" alt="image" src="https://github.com/user-attachments/assets/cd6fe64c-65fa-4070-a8bc-a6390511d413" />

<img width="1902" height="934" alt="image" src="https://github.com/user-attachments/assets/fb4002e0-e0c2-49a4-ab1a-a380465d2c6a" />





## 📊 Performance

- Optimized for clear, frontal face images
- Works with various lighting conditions
- Supports JPG, PNG, WEBP formats

## 🎓 Applications

- Customer sentiment analysis
- Mental health monitoring  
- User experience research
- Interactive entertainment
- Educational tools

## ⚠️ Limitations

- Requires visible face in image
- Best with frontal face poses
- Performance depends on image quality
- Currently supports 3 basic emotions

## 🛠️ Technical Stack

- **Gradio**: Web interface
- **PyTorch**: Deep learning framework
- **MTCNN**: Face detection
- **Hugging Face**: Deployment platform

## 📝 License

MIT License - Feel free to use for your projects!

## 👨‍💻 Developer

Created with ❤️ for facial emotion recognition

---

**Try it now!** Upload an image and see the magic happen ✨
```

---

### 4. **Create `.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
.ipynb_checkpoints

# Model files (will be downloaded)
*.pt
*.pth
*.onnx
emotion_vit_model.pt
```

---



