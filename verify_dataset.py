import os

base = "data/chest_xray"
for split in ["train", "test", "val"]:
    print(f"\n{split.upper()}")
    print("NORMAL:", len(os.listdir(f"{base}/{split}/NORMAL")))
    print("PNEUMONIA:", len(os.listdir(f"{base}/{split}/PNEUMONIA")))
