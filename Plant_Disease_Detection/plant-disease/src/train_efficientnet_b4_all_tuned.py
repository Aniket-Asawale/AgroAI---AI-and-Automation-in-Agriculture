import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler
from pathlib import Path
import os

# ---------------- CONFIG ----------------
DATA_ROOT = Path(r"C:\Users\jayes\plant_disease\data_cls\images")
BATCH_SIZE = 4
ACCUM_STEPS = 4          # effective batch size = 16
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 10
IMG_SIZE = 320
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = r"C:\Users\jayes\plant_disease\models\efficientnet_b4_tuned_final.pth"

torch.manual_seed(42)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)

print("Device:", DEVICE)

# ---------------- TRANSFORMS ----------------
weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------- RUN EPOCH ----------------
def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total, correct, running_loss = 0, 0, 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        if is_train:
            optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # ---- forward ----
            if scaler is not None:
                with autocast("cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    if is_train:
                        loss = loss / ACCUM_STEPS
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if is_train:
                    loss = loss / ACCUM_STEPS

            # ---- backward ----
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % ACCUM_STEPS == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * imgs.size(0) * (ACCUM_STEPS if is_train else 1)
            preds = outputs.argmax(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return running_loss / total, correct / total

# ---------------- MAIN ----------------
def main():
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print("num_classes:", num_classes)

    # ---------------- MODEL ----------------
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(DEVICE)

    # ---------------- CLASS WEIGHTS ----------------
    class_to_idx = train_ds.class_to_idx
    class_weights = torch.ones(num_classes, device=DEVICE)

    weak_names = ["wheat_black_rust", "wheat_brown_rust", "wheat_tan_spot"]
    for name in weak_names:
        if name in class_to_idx:
            class_weights[class_to_idx[name]] = 2.0

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = GradScaler("cuda") if DEVICE.type == "cuda" else None
    best_val_acc = 0.0

    # ---------------- STAGE 1 ----------------
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=1e-4,
    )

    print("Stage 1: training classifier head")
    for epoch in range(EPOCHS_STAGE1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"[Stage1] Epoch {epoch+1}/{EPOCHS_STAGE1} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(Path(SAVE_PATH).parent, exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model (Stage1): {val_acc:.4f}")

    # ---------------- STAGE 2 ----------------
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_STAGE2
    )

    print("Stage 2: fine-tuning full model")
    for epoch in range(EPOCHS_STAGE2):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"[Stage2] Epoch {epoch+1}/{EPOCHS_STAGE2} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model (Stage2): {val_acc:.4f}")

    # ---------------- TEST ----------------
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    t_total, t_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()

    print(f"TUNED EfficientNet-B4 Test accuracy: {t_correct / t_total:.4f}")

if __name__ == "__main__":
    main()
