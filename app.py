import os
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

# ---------------- UI ----------------
st.set_page_config(page_title="Chest X-Ray Diagnosis", layout="centered")
st.title("Chest X-Ray Diagnosis (VLM Baseline)")

# ---------------- Device ----------------
device = "cpu"

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 2)

    model_path = os.path.join("models", "baseline_model.pth")
    if not os.path.exists(model_path):
        st.error("❌ Model file not found: models/baseline_model.pth")
        st.stop()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# ---------------- Image Transform ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- File Upload ----------------
file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item() * 100

    if pred == 1:
        st.error(f"Diagnosis: PNEUMONIA ({confidence:.2f}%)")
        st.write("🩺 Recommendation: Consult a doctor for further evaluation.")
    else:
        st.success(f"Diagnosis: NORMAL ({confidence:.2f}%)")
