import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from PIL import Image

st.title("Chest X-Ray Diagnosis (VLM Baseline)")

device = "cpu"

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)
MODEL_PATH = os.path.join("models", "baseline_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded X-ray")

    x = transform(img).unsqueeze(0)
    pred = torch.argmax(model(x)).item()

    if pred == 1:
        st.error("Diagnosis: PNEUMONIA")
        st.write("Recommendation: Consult doctor, further tests required")
    else:
        st.success("Diagnosis: NORMAL")
