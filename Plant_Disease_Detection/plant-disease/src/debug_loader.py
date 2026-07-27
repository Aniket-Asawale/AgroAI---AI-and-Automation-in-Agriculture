import torch
from torchvision import datasets, transforms
from pathlib import Path

DATA_ROOT = Path(r"C:\Users\jayes\plant_disease\data_cls\images")
IMG_SIZE = 380
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05,
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def main():
    ds = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tf)
    print("Train size:", len(ds))
    img, label = ds[0]
    print("One sample shape:", img.shape, "label:", label)

if __name__ == "__main__":
    main()
