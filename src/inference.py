import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F

# Class mapping (VERY IMPORTANT)
classes = ["NORMAL", "PNEUMONIA"]

device = "cpu"

# Load model
model = resnet18()
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
model.eval()

# CORRECT transforms (with normalization)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
img = Image.open("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg").convert("RGB")
x = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs).item()

print("Prediction:", classes[pred_idx])
print("Confidence:", probs[0][pred_idx].item())
