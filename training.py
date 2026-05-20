import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import DR_EfficientNet_CBAM
from data_loader import get_dataloaders

# ---------------- CONFIG ----------------
EPOCHS = 6
BATCH_SIZE = 8
LR = 3e-4
NUM_CLASSES = 5
# ---------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader, val_loader = get_dataloaders(BATCH_SIZE)

# ---------- CLASS WEIGHTS (MANDATORY) ----------
class_counts = torch.tensor([5127, 549, 1107, 199, 189], dtype=torch.float)
weights = (1.0 / class_counts)
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights.to(device))

model = DR_EfficientNet_CBAM(NUM_CLASSES).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

best_acc = 0.0


def train_one_epoch():
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return loss_sum / len(train_loader), correct / total


def validate():
    model.eval()
    correct, total, loss_sum = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(val_loader), correct / total


# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = validate()

    print(f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "efficientnet_cbam_best.pth")
        print("✅ Best model saved")

print("\n🎉 Training Done")
print("Best Validation Accuracy:", best_acc * 100)
