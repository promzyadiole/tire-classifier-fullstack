import io
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pth"


device = torch.device("cpu")

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_model() -> nn.Module:
    resnet50_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )

    resnet50_model.fc = nn.Identity()
    resnet50_model = resnet50_model.to(device)

    fc_model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1),
    )

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    fc_model.load_state_dict(state_dict)
    fc_model = fc_model.to(device)

    model = nn.Sequential(
        resnet50_model,
        fc_model,
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


def predict_tire(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(dim=0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        y_pred = torch.sigmoid(model(image_tensor))
        y_pred_value = y_pred.item()
        percentage = round(y_pred_value * 100, 3)

    if percentage > 50:
        label = "Good tire"
        recommendation = "You can ride."
        status = "safe"
    else:
        label = "Bad tire"
        recommendation = "Avoid riding for safety."
        status = "danger"

    return {
        "confidence": percentage,
        "label": label,
        "status": status,
        "recommendation": recommendation,
        "message": f"Confidence: {percentage}% — {label}, {recommendation}",
    }