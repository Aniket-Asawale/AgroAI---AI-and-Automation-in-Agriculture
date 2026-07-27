import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import os
import math

# ===================== CONFIG =====================
CROP_NAME = "WHEAT"
DATA_ROOT = Path(r"C:\Users\jayes\plant_disease\data_wheat\images")
SAVE_PATH = r"C:\Users\jayes\plant_disease\models\efficientnet_b4_wheat_tuned.pth"

BATCH_SIZE = 8          # increase or use grad accumulation if GPU allows
EPOCHS = 30             # total epochs (5 warmup + 25 finetune)
WARMUP_EPOCHS = 5       # classifier-only training
LR_HEAD = 3e-4          # LR for warmup
LR_FULL = 1e-4          # LR for full finetuning
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ===================== TRANSFORMS =====================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.02),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ===================== LOSS =====================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.07):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = nn.functional.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * logprobs, dim=-1))

# ===================== MAIN =====================
def main():
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print(f"{CROP_NAME} num_classes: {num_classes}")
    print("Train batches:", len(train_loader))

    # ===================== MODEL =====================
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(DEVICE)

    criterion = LabelSmoothingCE(smoothing=0.07)

    # ===================== STAGE 1: WARMUP =====================
    print("\n===== Stage 1: Classifier warm-up =====")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR_HEAD, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(WARMUP_EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if batch_idx == 0 or batch_idx % 50 == 0:
                print(f"[{CROP_NAME}] Epoch {epoch+1} batch {batch_idx}")

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        train_loss /= total

        val_acc = evaluate(model, val_loader, criterion)
        print(f"[{CROP_NAME}] Warmup Epoch {epoch+1}/{WARMUP_EPOCHS} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save(model, train_ds, epoch+1, val_acc)

    # ===================== STAGE 2: FULL FINETUNE =====================
    print("\n===== Stage 2: Full fine-tuning =====")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    total_steps = len(train_loader) * (EPOCHS - WARMUP_EPOCHS)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for epoch in range(WARMUP_EPOCHS, EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            if batch_idx == 0 or batch_idx % 50 == 0:
                print(f"[{CROP_NAME}] Epoch {epoch+1} batch {batch_idx}")

            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total
        train_loss /= total

        val_acc = evaluate(model, val_loader, criterion)
        print(f"[{CROP_NAME}] Epoch {epoch+1}/{EPOCHS} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save(model, train_ds, epoch+1, val_acc)

    # ===================== TEST =====================
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_acc = evaluate(model, test_loader, None, test=True)
    print(f"[{CROP_NAME}] Test accuracy: {test_acc:.4f}")

# ===================== HELPERS =====================
def evaluate(model, loader, criterion=None, test=False):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


def save(model, train_ds, epoch, val_acc):
    os.makedirs(Path(SAVE_PATH).parent, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": train_ds.class_to_idx,
        "epoch": epoch,
        "val_acc": val_acc,
    }, SAVE_PATH)
    print(f"Saved best {CROP_NAME} model: val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
