from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from pathlib import Path

class BasketballPositionDataset(Dataset):
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = Path(images_dir)
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        if self.data.empty:
            raise ValueError("El CSV no contiene datos.")

        # Leer tamaño de la primera imagen para normalizar coordenadas
        sample_image_path = self.images_dir / self.data.iloc[0]['image']
        with Image.open(sample_image_path) as img:
            self.width, self.height = img.size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = self.images_dir / row['image']
        image = Image.open(image_path).convert("RGB")

        # Coordenadas normalizadas
        x = float(row['x']) / self.width
        y = float(row['y']) / self.height
        target = torch.tensor([x, y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target
