import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import gradio as gr
import numpy as np
import cv2
import json

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

MODEL_PATHS = {
    "All crops (30 classes)": MODELS_DIR / "efficientnet_b4_all_final.pth",
    "Corn": MODELS_DIR / "efficientnet_b4_corn.pth",
    "Rice": MODELS_DIR / "efficientnet_b4_rice.pth",
    "Wheat": MODELS_DIR / "efficientnet_b4_wheat_tuned.pth",
    "Millet": MODELS_DIR / "efficientnet_b4_millet.pth",
    "Sugarcane": MODELS_DIR / "efficientnet_b4_sugarcane.pth",
}

KB_PATH = BASE_DIR / "disease_kb.json"
IMG_SIZE = 380
MODEL_TEST_ACCURACY = 0.836

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ===================== LOAD KNOWLEDGE BASE =====================
with open(KB_PATH, "r", encoding="utf-8") as f:
    disease_kb = json.load(f)

# ===================== TRANSFORMS =====================
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ===================== LOAD MODEL =====================
def load_single_model(path: Path):
    ckpt = torch.load(path, map_location=DEVICE)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(class_to_idx)
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    return model, idx_to_class

models_dict, idx_to_class_dict = {}, {}
for name, path in MODEL_PATHS.items():
    m, idx2cls = load_single_model(path)
    models_dict[name] = m
    idx_to_class_dict[name] = idx2cls

# ===================== GRAD-CAM =====================
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._forward)
        layer.register_full_backward_hook(self._backward)

    def _forward(self, m, i, o):
        self.activations = o

    def _backward(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, class_idx].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

cam_extractors = {
    name: GradCAM(m, m.features[-1]) for name, m in models_dict.items()
}

# ===================== CONSISTENCY BADGE =====================
def consistency_badge(text):
    if "High" in text:
        c = "#4caf50"
    elif "Medium" in text:
        c = "#ff9800"
    elif "Low" in text:
        c = "#f44336"
    else:
        c = "#9e9e9e"

    return f"""
    <div style="
        background:{c};
        color:white;
        padding:8px;
        border-radius:8px;
        font-weight:600;
        text-align:center;
    ">{text}</div>
    """

# ===================== DISEASE INFO =====================
def disease_cards(key, conf):
    if key not in disease_kb:
        return "<p style='color:red'>No disease data found.</p>"

    d = disease_kb[key]
    color = "#4caf50" if conf > 80 else "#ff9800" if conf > 50 else "#f44336"

    def ul(items):
        return "".join(f"<li>{i}</li>" for i in items)

    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px">

    <div style="background:#1e1e1e;padding:16px;border-radius:12px">
        <h3>Overview</h3>
        <p><b>Crop:</b> {d['crop']}</p>
        <p><b>Disease:</b> {key.replace('_',' ').title()}</p>
        <p><b>Type:</b> {d['type']}</p>
        <span style="background:{color};padding:4px 10px;border-radius:6px">
        Confidence: {conf:.2f}%
        </span>
        <div style="background:#555;height:12px;border-radius:6px;margin-top:8px">
            <div style="width:{int(conf)}%;background:{color};height:100%;border-radius:6px"></div>
        </div>
    </div>

    <div style="background:#1e1e1e;padding:16px;border-radius:12px">
        <h3>Symptoms</h3>
        <ul>{ul(d['symptoms'])}</ul>
    </div>

    <div style="background:#222831;padding:16px;border-radius:12px">
        <h3> Chemical Treatment</h3>
        <p><b>Medicine:</b> {d['chemical_treatment']['medicine']}</p>
        <p><b>Dosage:</b> {d['chemical_treatment']['dosage']}</p>
    </div>

    <div style="background:#1b2d1b;padding:16px;border-radius:12px">
        <h3> Organic Treatment</h3>
        <p><b>Medicine:</b> {d['organic_treatment']['medicine']}</p>
        <p><b>Dosage:</b> {d['organic_treatment']['dosage']}</p>
    </div>


    <div style="background:#2a1f1f;padding:16px;border-radius:12px">
        <h3>Precautions</h3>
        <ul>{ul(d['precautions'])}</ul>
    </div>

    </div>
    """

# ===================== PREDICT =====================
def predict(img, crop, show_cam):
    model = models_dict[crop]
    idx2cls = idx_to_class_dict[crop]
    cam_ext = cam_extractors[crop]

    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
        conf, idx = torch.max(p, 0)

    pred = idx2cls[idx.item()]
    conf = round(conf.item() * 100, 2)

    all_model = models_dict["All crops (30 classes)"]
    all_idx2cls = idx_to_class_dict["All crops (30 classes)"]
    with torch.no_grad():
        all_pred = all_idx2cls[
            torch.argmax(torch.softmax(all_model(x), 1)).item()
        ]

    this_crop = pred.split("_")[0].capitalize()
    auto_crop = all_pred.split("_")[0].capitalize()

    if crop == "All crops (30 classes)":
        consistency = "Not applicable"
    elif this_crop == crop and auto_crop == crop:
        consistency = f"High (all models agree on {crop})"
    elif this_crop == crop:
        consistency = f"Medium (disease: {crop}, all-crop: {auto_crop})"
    else:
        consistency = f"Low (selected: {crop}, disease: {this_crop})"

    cam_img = None
    if show_cam:
        cam = cam_ext.generate(x, idx.item())[0].detach().cpu().numpy()
        cam = cv2.resize(cam, img.size)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = Image.fromarray(cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0))

    return (
        pred,
        conf,
        MODEL_TEST_ACCURACY * 100,
        cam_img,
        disease_cards(pred, conf),
        consistency_badge(consistency),
    )

# ===================== UI =====================
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center'> Plant Disease Detection & Recommendation</h1>")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", height=350, label="Upload Leaf Image")
            crop_dd = gr.Dropdown(list(MODEL_PATHS.keys()), value="All crops (30 classes)")
            cam_chk = gr.Checkbox(value=True, label="Show Grad-CAM")
            btn = gr.Button("Analyze")

        with gr.Column():
            out_pred = gr.Textbox(label="Predicted Disease")
            out_conf = gr.Number(label="Confidence (%)")
            out_acc = gr.Number(label="Model Test Accuracy (%)")
            out_cons = gr.HTML(label="Crop Consistency")

    gr.Markdown("### Model Attention (Grad-CAM)")
    out_cam = gr.Image(height=350)

    gr.Markdown("""4
## Disease Information & Treatment
<p style="font-size:15px;color:#b0b0b0;margin-top:-8px">
⚠️ Consult local agricultural officer before applying treatment
</p>
""")

    out_info = gr.HTML()

    btn.click(
        predict,
        [img_in, crop_dd, cam_chk],
        [out_pred, out_conf, out_acc, out_cam, out_info, out_cons],
    )

if __name__ == "__main__":
    demo.launch()
