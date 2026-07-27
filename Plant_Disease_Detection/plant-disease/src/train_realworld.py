import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        input_size = 380
        return model, input_size
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        input_size = 224
        return model, input_size
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_size = 224
        return model, input_size
    raise ValueError(f"Unsupported arch: {arch}")


def build_transforms(img_size: int):
    # Stronger augmentations for realistic farmer-captured photos.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.75, 1.35)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.35),
            transforms.RandomAutocontrast(p=0.25),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def compute_class_weights(train_ds) -> torch.Tensor:
    counts = torch.zeros(len(train_ds.classes), dtype=torch.float)
    for _, cls_idx in train_ds.samples:
        counts[cls_idx] += 1
    counts = torch.clamp(counts, min=1.0)
    # Balanced weighting: N / (K * n_i)
    total = counts.sum()
    k = len(train_ds.classes)
    weights = total / (k * counts)
    return weights


def run_epoch(model, loader, criterion, device, optimizer=None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train robust plant-disease models for field photos.")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root containing train/val/test.")
    parser.add_argument("--arch", type=str, default="efficientnet_b4", choices=["efficientnet_b4", "efficientnet_b0", "resnet18"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path (.pth)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "train").exists():
        raise FileNotFoundError(f"Missing train folder: {data_root / 'train'}")
    if not (data_root / "val").exists():
        raise FileNotFoundError(f"Missing val folder: {data_root / 'val'}")
    if not (data_root / "test").exists():
        raise FileNotFoundError(f"Missing test folder: {data_root / 'test'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build temporary model to get size.
    temp_model, img_size = build_model(args.arch, num_classes=2)
    del temp_model
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise ValueError("Class mappings differ between train/val/test. Ensure identical class folder names.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model, _ = build_model(args.arch, num_classes=len(train_ds.classes))
    model = model.to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "val_acc": float(val_acc),
                    "arch": args.arch,
                    "input_size": img_size,
                },
                output_path,
            )
            print(f"Saved best checkpoint: {output_path} (val_acc={val_acc:.4f})")

    ckpt = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_acc = evaluate(model, test_loader, device)
    print(f"Best val_acc={best_val:.4f} | test_acc={test_acc:.4f}")

    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": args.arch,
                "input_size": img_size,
                "classes": train_ds.classes,
                "num_classes": len(train_ds.classes),
                "best_val_acc": best_val,
                "test_acc": test_acc,
                "data_root": str(data_root),
            },
            f,
            indent=2,
        )
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

