import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_ds = ImageFolder("data/chest_xray/train", transform=transform)
test_ds  = ImageFolder("data/chest_xray/test", transform=transform)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=16)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 2   # keep it small for now

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} completed, Avg Loss: {running_loss/len(train_dl):.4f}")

torch.save(model.state_dict(), "models/baseline_model.pth")
print("Baseline model saved successfully")
