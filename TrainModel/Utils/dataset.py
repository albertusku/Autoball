from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch
from pathlib import Path

class BasketballPositionDataset(Dataset):
    def __init__(self, labels_df, transform=None):
        self.data = labels_df
        self.transform = transform

        if self.data.empty:
            raise ValueError("El CSV no contiene datos.")

        # Leer tamaño real de la primera imagen
        sample_path = Path(self.data.iloc[0]["image"])
        with Image.open(sample_path) as img:
            self.width, self.height = img.size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = Path(row['image']) 
        image = Image.open(image_path).convert("RGB")

        # Coordenadas normalizadas
        x = float(row['x']) / self.width
        y = float(row['y']) / self.height
        target = torch.tensor([x, y], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def load_all_labels(labels_root="Labels", extracted_root="ExtractedFrames"):
    labels_root = Path(labels_root)
    extracted_root = Path(extracted_root)
    video_dirs = [d for d in labels_root.iterdir() if d.is_dir()]
    
    all_data = []
    for video_dir in video_dirs:
        csv_path = video_dir / "labels.csv"
        if not csv_path.exists():
            print(f"[AVISO] No se encontró: {csv_path}, se omite.")
            continue

        df = pd.read_csv(csv_path)
        video_name = video_dir.name
        df["video"] = video_name
        df["image"] = str(extracted_root / video_name) + "/" + df["image"]
        all_data.append(df)

    if not all_data:
        raise ValueError("No se cargó ningún label. Verifica las carpetas.")

    return pd.concat(all_data, ignore_index=True)
