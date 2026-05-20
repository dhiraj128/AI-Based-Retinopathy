import os
from PIL import Image

INPUT_DIR = "Datasets/Final_DR_Dataset"
OUTPUT_DIR = "Datasets/Preprocessed_DR_Dataset"
IMAGE_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

processed = 0
removed = 0

for cls in os.listdir(INPUT_DIR):
    in_cls = os.path.join(INPUT_DIR, cls)
    out_cls = os.path.join(OUTPUT_DIR, cls)

    if not os.path.isdir(in_cls):
        continue

    os.makedirs(out_cls, exist_ok=True)

    for img_name in os.listdir(in_cls):
        img_path = os.path.join(in_cls, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            img.save(os.path.join(out_cls, img_name))
            processed += 1
        except Exception:
            removed += 1
            print("Removed corrupted:", img_path)

print("\n✅ PREPROCESSING COMPLETED")
print("Processed images:", processed)
print("Removed corrupted images:", removed)
