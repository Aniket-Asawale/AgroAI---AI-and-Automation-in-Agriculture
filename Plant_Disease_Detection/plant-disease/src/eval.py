import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# -------------------- Paths --------------------
DATA_ROOT = Path(r"C:\Users\jayes\plant_disease\data_millet\images")
MODEL_PATH = Path(r"C:\Users\jayes\plant_disease\models\efficientnet_b4_millet.pth")
OUTPUT_DIR = Path(r"C:\Users\jayes\plant_disease\eval_outputs")

# -------------------- Config --------------------
IMG_SIZE = 320
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -------------------- Transforms --------------------
test_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------- Load Model --------------------
def load_model(num_classes: int):
    print(f"Loading model from: {MODEL_PATH}")

    ckpt = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False  # REQUIRED for PyTorch 2.6+
    )

    # Load EfficientNet-B4 backbone
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    return model

# -------------------- Main --------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = datasets.ImageFolder(DATA_ROOT / "test", transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Loaded test dataset with {len(test_ds)} images")
    print(f"Number of classes: {len(test_ds.classes)}")

    model = load_model(num_classes=len(test_ds.classes))

    all_preds = []
    all_labels = []

    # -------------------- Inference --------------------
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # -------------------- Metrics --------------------
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc:.4f}")

    # -------------------- Classification Report --------------------
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=test_ds.classes,
        digits=4,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).T
    report_path = OUTPUT_DIR / "classification_report_efficientnet_b4_millet.csv"
    report_df.to_csv(report_path)
    print(f"Saved classification report → {report_path}")

    # -------------------- Confusion Matrix --------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12), dpi=120)
    sns.set(font_scale=0.8)
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=test_ds.classes,
        yticklabels=test_ds.classes,
        square=True,
        cbar=True,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – EfficientNet-B4 (Millet)")
    plt.tight_layout()

    cm_path = OUTPUT_DIR / "confusion_matrix_efficientnet_b4_millet.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"Saved confusion matrix → {cm_path}")

    # -------------------- Correlation Matrix --------------------
    print("Generating correlation matrix...")

    # Normalize confusion matrix (row-wise)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # Convert to DataFrame
    cm_df = pd.DataFrame(
        cm_normalized,
        index=test_ds.classes,
        columns=test_ds.classes
    )

    # Compute correlation matrix
    corr_matrix = cm_df.corr(method="pearson")

    # Save CSV
    corr_csv_path = OUTPUT_DIR / "correlation_matrix_efficientnet_b4_millet.csv"
    corr_matrix.to_csv(corr_csv_path)

    # Plot heatmap
    plt.figure(figsize=(12, 10), dpi=120)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=test_ds.classes,
        yticklabels=test_ds.classes,
        square=True,
        cbar=True
    )

    plt.title("Class Correlation Matrix – EfficientNet-B4 (Millet)")
    plt.tight_layout()

    corr_img_path = OUTPUT_DIR / "correlation_matrix_efficientnet_b4_millet.png"
    plt.savefig(corr_img_path)
    plt.close()

    print(f"Saved correlation CSV → {corr_csv_path}")
    print(f"Saved correlation plot → {corr_img_path}")

    print("\nEvaluation complete.")

# -------------------- Run --------------------
if __name__ == "__main__":
    main()
