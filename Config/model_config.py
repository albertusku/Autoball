from torchvision import models, transforms
import torch.nn as nn
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_config = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])


model_config = models.resnet50(pretrained=True)
model_config.fc = nn.Sequential(
    nn.Linear(model_config.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
) 
model_config = model_config.to(DEVICE)

