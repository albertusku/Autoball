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
import argparse
import matplotlib.patches as patches

# Configuración
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def  compute_pixel_errors(model, dataset, device,see_points=False):
    tolerance = 15  # Tolerancia en píxeles
    model.eval()
    pixel_errors = []
    list_coords_pred = []
    list_coords_gt = []

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
        list_coords_pred.append((x_pred, y_pred))
        list_coords_gt.append((x_gt, y_gt))

        error = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
        pixel_errors.append(error)
    if args.see_points:
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
        ax.set_xlim(0, 640)
        ax.set_ylim(360, 0)  # invertir eje Y para que (0,0) esté arriba a la izquierda
        ax.set_title("Predicciones vs. Reales (360p)")
        ax.set_facecolor('black')

        for x_pred, y_pred in list_coords_pred:
            ax.add_patch(patches.Circle((x_pred, y_pred), radius=4, color='lime'))

        # Dibujar puntos reales en rojo
        for x_gt, y_gt in list_coords_gt:
            ax.add_patch(patches.Circle((x_gt, y_gt), radius=4, color='red'))
        
        output_path = "/home/ruiz17/Autoball/TrainModel/Model/pred_vs_gt_360p.png"
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    pixel_errors = np.array(pixel_errors)
    print(f"\nError medio: {pixel_errors.mean():.2f} px")
    print(f"Mediana: {np.median(pixel_errors):.2f} px")
    print(f"Máximo: {pixel_errors.max():.2f} px")
    print(f"≤{tolerance}px: {(pixel_errors <= tolerance).sum()} / {len(pixel_errors)} imágenes ({(pixel_errors <= tolerance).mean()*100:.1f}%)")

    pixel_errors = np.array(pixel_errors)
    accepted = (pixel_errors <= tolerance).sum()
    total = len(pixel_errors)
    percentage = 100 * accepted / total

    return percentage

def main(args):

    # Transformaciones
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
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
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )  # Salida: x, y
    model = model.to(DEVICE)

    # Pérdida y optimizador
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)


    # Entrenamiento
    train_losses, val_losses = [], []

    if args.check_error:
        print("Calculando errores de píxeles en el dataset...")
        model.load_state_dict(torch.load("Model/Autoball_model.pth", map_location=DEVICE))
        compute_pixel_errors(model, dataset, DEVICE)
        return

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
        scheduler.step(avg_val_loss)


        print(f" Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


    #  Plot de pérdidas
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Loss durante entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.savefig("Model/loss_plot.png")

    accepted_tolerance=compute_pixel_errors(model, dataset, DEVICE,args.see_points)
    try:
        with open("Model/accepted_tolerance.txt", "r") as f:
            old_tolerance = float(f.read().strip())
    except FileNotFoundError:
        old_tolerance = 0.0
    if accepted_tolerance > old_tolerance:
        print("El modelo ha mejorado su precisión, guardando nuevo modelo.")
        os.makedirs("Model", exist_ok=True)
        torch.save(model.state_dict(), "Model/Autoball_model.pth")
        with open("Model/accepted_tolerance.txt", "w") as f:
            f.write(f"{accepted_tolerance:.2f}")
    else:
        print("El modelo no ha mejorado su precisión, no se guarda el nuevo modelo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script para extracción de frames y anotación manual del balón"
    )

    parser.add_argument(
        "--check_error",
        action="store_true",
        help="Calcula el error medio de las predicciones del modelo en píxeles",
    )

    parser.add_argument(
        "--see_points",
        action="store_true",
        help="Visualiza las coordenadas predichas y reales en un gráfico",
    )


    args = parser.parse_args()
    main(args)
