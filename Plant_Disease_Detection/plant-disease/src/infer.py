import torch
from torch import nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(r"C:\Users\jayes\plant_disease\models\efficientnet_b4_final.pth")
IMG_SIZE = 380

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_to_idx))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class

def predict_image(img_path: str, model, idx_to_class):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    pred_class = idx_to_class[pred_idx.item()]
    return pred_class, conf.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python infer.py path/to/image.jpg")
        raise SystemExit

    img_path = sys.argv[1]
    model, idx_to_class = load_model()
    pred_class, conf = predict_image(img_path, model, idx_to_class)
    print(f"Prediction: {pred_class} (confidence {conf:.3f})")
