import torch
import random
from torchvision import transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from Utils.dataset import BasketballPositionDataset

# Configuración
MODEL_PATH = "Model/Autoball_model.pth"
IMAGES_DIR = "ExtractedFrames/test1"
LABELS_CSV = "Labels/test1/labels.csv"
N_SAMPLES = 5
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
dataset = BasketballPositionDataset(IMAGES_DIR, LABELS_CSV, transform=transform)

# Modelo
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Seleccionar imágenes aleatorias
indices = random.sample(range(len(dataset)), N_SAMPLES)


for idx in indices:
    img_tensor, target = dataset[idx]
    img_path = os.path.join(IMAGES_DIR, dataset.data.iloc[idx]['image'])

    with Image.open(img_path) as original_img:
        w, h = original_img.size
        input_img = img_tensor.unsqueeze(0).to(DEVICE)
        pred = model(input_img).squeeze().cpu().detach().numpy()

        # Desnormalizar coordenadas
        x_pred, y_pred = int(pred[0] * w), int(pred[1] * h)
        x_true, y_true = int(target[0] * w), int(target[1] * h)

        # Crear figura
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(original_img)
        ax.set_title(f"GT: ({x_true},{y_true})\nPred: ({x_pred},{y_pred})")
        ax.add_patch(patches.Circle((x_true, y_true), radius=10, color='red', fill=False, linewidth=2))
        ax.add_patch(patches.Circle((x_pred, y_pred), radius=10, color='lime', fill=False, linewidth=2))
        ax.axis('off')

        plt.tight_layout()
        plt.show()  # Espera a que cierres antes de continuar

