

import os
import csv
import timm
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


# константы 
NUM_CLASSES      = 52                 # число классов (количество типов карт, 0 - 52, где 52 - отсутствие )
BATCH_SIZE       = 16                 
LR               = 1e-4
EPOCHS           = 100
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'


# для работы в jupyter:

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
else:
    ROOT = Path.cwd().resolve().parents[0] # jup

TRAIN_DIR        = ROOT / "dataset/eff_class/train"
VAL_DIR          = ROOT / "dataset/eff_class/val"
OUTPUT_WEIGHTS   = ROOT / "pretrained" / "efficientnetb0_best.pth"

# для создания данных об обучении 
RUN_DIR = os.path.join("train_logs", "efficientnet",
                       datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
CSV_PATH = os.path.join(RUN_DIR, "train_log.csv")
PLOT_PATH = os.path.join(RUN_DIR, "curves.png")

# Augm
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=(0.485,0.456,0.406),
                         std=(0.229,0.224,0.225)),
])



train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=train_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


best_val_acc = 0.0

history = {
    "epoch":[],
    "train_loss":[],
    "train_acc": [],
    "val_acc": []
}



for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()

    epoch_loss = running_loss / len(train_ds)
    epoch_acc  = running_corrects / len(train_ds)

    model.eval()
    val_corrects = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_corrects += (preds == labels).sum().item()

    val_acc = val_corrects / len(val_ds)

    history["epoch"].append(epoch)
    history['train_loss'].append(epoch_loss)
    history["train_acc"].append(epoch_acc)
    history["val_acc"].append(val_acc)

    print(f'Epoch {epoch}/{EPOCHS} | '
          f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | '
          f'Val Acc: {val_acc:.4f}')

    # best weights
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
        torch.save(model.state_dict(), OUTPUT_WEIGHTS)
        print(f'Сохранили лучшеи веса (val_acc={val_acc:.4f})')

# сохранение логов
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "train_acc", "val_acc"])
    for e, tl, ta, va in zip(history["epoch"], history["train_loss"],
                             history["train_acc"], history["val_acc"]):
        w.writerow([e, tl, ta, va])


plt.figure()
plt.plot(history["epoch"], history["train_loss"], label="train_loss")
plt.plot(history["epoch"], history["train_acc"],  label="train_acc")
plt.plot(history["epoch"], history["val_acc"],    label="val_acc")
plt.xlabel("epoch"); plt.ylabel("value"); plt.title("EfficientNet training")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.close()

print(f"[EfficientNet] Логи сохранены в {RUN_DIR}")

