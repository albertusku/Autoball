import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from Utils.dataset import BasketballPositionDataset, load_all_labels
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Configuración
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def  compute_pixel_errors(model, dataset, device):
    model.eval()
    pixel_errors = []

    for idx in tqdm(range(len(dataset)), desc="Evaluando"):
        img_tensor, target = dataset[idx]
        img_path = Path(dataset.data.iloc[idx]['image'])

        with Image.open(img_path) as img:
            w, h = img.size

        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().cpu().numpy()

        # Coordenadas en píxeles
        x_gt = target[0].item() * w
        y_gt = target[1].item() * h
        x_pred = pred[0] * w
        y_pred = pred[1] * h

        error = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
        pixel_errors.append(error)

    pixel_errors = np.array(pixel_errors)
    print(f"\n📊 Error medio: {pixel_errors.mean():.2f} px")
    print(f"📉 Mediana: {np.median(pixel_errors):.2f} px")
    print(f"📈 Máximo: {pixel_errors.max():.2f} px")
    print(f"✅ ≤10px: {(pixel_errors <= 10).sum()} / {len(pixel_errors)} imágenes ({(pixel_errors <= 10).mean()*100:.1f}%)")
    return pixel_errors

def main():

    # Transformaciones
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Dataset y splits
    labels_df = load_all_labels()
    labels_df.to_csv("Labels/combined_labels.csv", index=False)
    dataset = BasketballPositionDataset(labels_df, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Modelo
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Salida: x, y
    model = model.to(DEVICE)

    # Pérdida y optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Entrenamiento
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f" Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    #  Guardar modelo
    os.makedirs("Model", exist_ok=True)
    torch.save(model.state_dict(), "Model/Autoball_model.pth")

    #  Plot de pérdidas
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss durante entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.savefig("Model/loss_plot.png")

    compute_pixel_errors(model, dataset, DEVICE)

if __name__ == "__main__":
    main()
