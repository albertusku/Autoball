from torchvision import models, transforms
import torch.nn as nn
import torch
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Model/Autoball_model.pth"  # Ruta al modelo entrenado

class HybridLoss(nn.Module):
    def __init__(self, img_width=640, img_height=360, margin_px=15, penalty_factor=3.0):
        super().__init__()
        self.base_loss = nn.SmoothL1Loss()
        self.margin = margin_px / np.sqrt(img_width**2 + img_height**2)  # Normalizado
        self.penalty_factor = penalty_factor

    def forward(self, pred, target):
        # pred y target están normalizados en [0, 1]
        base = self.base_loss(pred, target)
        
        # Distancia euclidiana por muestra
        d = torch.norm(pred - target, dim=1)
        
        # Penaliza si el error supera el margen
        penalty = torch.clamp(d - self.margin, min=0.0).mean()
        
        return base + self.penalty_factor * penalty


transform_config = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])


def get_model(for_training=True, load_weights=True, weights_path="Model/Autoball_model.pth"):
    model = models.resnet50(pretrained=for_training)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model = model.to(DEVICE)
    
    if not for_training:
        model.eval()
    
    if load_weights and not for_training:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    
    return model

