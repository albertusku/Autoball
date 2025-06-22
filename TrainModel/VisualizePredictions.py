import torch
import random
from torchvision import transforms
from torchvision.models import resnet18,resnet34,resnet50
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from Utils.dataset import BasketballPositionDataset, load_all_labels
from Config.model_config import get_model, transform_config

# Configuración
MODEL_PATH = "Model/Autoball_model.pth"
N_SAMPLES = 50
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels_df = load_all_labels()
dataset = BasketballPositionDataset(labels_df, transform=transform_config)

# Seleccionar imágenes aleatorias
indices = random.sample(range(len(dataset)), N_SAMPLES)

model_config = get_model(for_training=False, load_weights=True, weights_path=MODEL_PATH)
for idx in indices:
    img_tensor, target = dataset[idx]
    img_path = dataset.data.iloc[idx]['image']  # Ya es path completo

    with Image.open(img_path) as original_img:
        w, h = original_img.size
        input_img = img_tensor.unsqueeze(0).to(DEVICE)
        pred = model_config(input_img).squeeze().cpu().detach().numpy()

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

