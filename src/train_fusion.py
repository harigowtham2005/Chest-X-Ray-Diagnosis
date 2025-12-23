import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# --------------------
# DEVICE
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------
# TEXT DATA (SIMPLE & EXPLAINABLE)
# --------------------
text_map = {
    "NORMAL": "No symptoms routine check normal lungs",
    "PNEUMONIA": "Patient has cough fever chest pain infection"
}

vectorizer = TfidfVectorizer(max_features=20)
vectorizer.fit(text_map.values())

# --------------------
# IMAGE TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------
# DATASET
# --------------------
train_ds = ImageFolder("data/chest_xray/train", transform=transform)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

# --------------------
# MODEL (VISION + TEXT FUSION)
# --------------------
class FusionModel(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()          # 512 features
        self.fc = nn.Linear(512 + text_dim, 2)

    def forward(self, img, txt):
        img_feat = self.cnn(img)
        combined = torch.cat((img_feat, txt), dim=1)
        return self.fc(combined)

text_dim = len(vectorizer.get_feature_names_out())
model = FusionModel(text_dim).to(device)

# --------------------
# TRAINING SETUP
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 2   # THIS IS ENOUGH

# --------------------
# TRAIN LOOP
# --------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (imgs, labels) in enumerate(train_dl):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # create text features
        text_features = []
        for lbl in labels:
            cls_name = train_ds.classes[lbl]
            vec = vectorizer.transform([text_map[cls_name]]).toarray()[0]
            text_features.append(vec)

        text_features = torch.tensor(
            np.array(text_features), dtype=torch.float32
        ).to(device)

        optimizer.zero_grad()
        outputs = model(imgs, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed, Avg Loss: {running_loss/len(train_dl):.4f}")

# --------------------
# SAVE MODEL
# --------------------
torch.save(model.state_dict(), "models/fused_model.pth")
print("Fusion model saved successfully")
