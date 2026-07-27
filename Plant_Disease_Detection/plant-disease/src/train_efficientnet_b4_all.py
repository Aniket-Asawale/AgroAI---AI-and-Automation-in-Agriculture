import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import os

DATA_ROOT = Path(r"C:\Users\jayes\plant_disease\data_cls\images")
BATCH_SIZE = 8
EPOCHS = 15
LR = 5e-4
IMG_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = r"C:\Users\jayes\plant_disease\models\efficientnet_b4_final.pth"

print("Device:", DEVICE)

weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def main():
    train_ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_ROOT / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(DATA_ROOT / "test",  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(train_ds.classes)
    print("num_classes:", num_classes)

    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total, correct, train_loss = 0, 0, 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
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

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} – batch {batch_idx}")

        train_loss /= total
        train_acc = correct / total

        model.eval()
        v_total, v_correct, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        val_loss /= v_total
        val_acc = v_correct / v_total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(Path(SAVE_PATH).parent, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, SAVE_PATH)
            print(f"Saved best EfficientNet-B4 model: val_acc={val_acc:.4f}")

    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    t_total, t_correct = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            t_total += labels.size(0)
            t_correct += (preds == labels).sum().item()

    print(f"EfficientNet-B4 Test accuracy: {t_correct / t_total:.4f}")

if __name__ == "__main__":
    main()
