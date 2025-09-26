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
from Config.model_config import transform_config,get_model,HybridLoss
from Config.env_config import *


def get_points(list_coords_pred, list_coords_gt):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)
    ax.set_xlim(0, 640)
    ax.set_ylim(360, 0)  # Invert Y axis so that (0,0) is at the top left
    ax.set_title("Predicciones vs. Reales (360p)")
    ax.set_facecolor('black')

    for x_pred, y_pred in list_coords_pred:
        ax.add_patch(patches.Circle((x_pred, y_pred), radius=4, color='lime'))

    # Painting red circles for ground truth
    for x_gt, y_gt in list_coords_gt:
        ax.add_patch(patches.Circle((x_gt, y_gt), radius=4, color='red'))
    
    output_path = "/home/ruiz17/Autoball/TrainModel/Model/pred_vs_gt_360p.png"
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def  compute_pixel_errors(model, dataset, device,see_points=False):
    tolerance = 15  # pixel tolerance for acceptance
    model.eval()
    pixel_errors = []
    list_coords_pred = []
    list_coords_gt = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        img_tensor, target = dataset[idx]
        img_path = Path(dataset.data.iloc[idx]['image'])

        with Image.open(img_path) as img:
            w, h = img.size

        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).squeeze().cpu().numpy()

        # pixel coordinates
        x_gt = target[0].item() * w
        y_gt = target[1].item() * h
        x_pred = pred[0] * w
        y_pred = pred[1] * h
        list_coords_pred.append((x_pred, y_pred))
        list_coords_gt.append((x_gt, y_gt))

        error = np.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
        pixel_errors.append(error)
    if args.see_points:
        get_points(list_coords_pred, list_coords_gt)
        

    pixel_errors = np.array(pixel_errors)
    print(f"\nmean error: {pixel_errors.mean():.2f} px")
    print(f"median: {np.median(pixel_errors):.2f} px")
    print(f"max: {pixel_errors.max():.2f} px")
    print(f"≤{tolerance}px: {(pixel_errors <= tolerance).sum()} / {len(pixel_errors)} images ({(pixel_errors <= tolerance).mean()*100:.1f}%)")

    pixel_errors = np.array(pixel_errors)
    accepted = (pixel_errors <= tolerance).sum()
    total = len(pixel_errors)
    percentage = 100 * accepted / total

    return percentage

def main(args):


    # Dataset and splits
    labels_df = load_all_labels()
    labels_df.to_csv("Labels/combined_labels.csv", index=False)
    dataset = BasketballPositionDataset(labels_df, transform=transform_config)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    num_workers = os.cpu_count()  # Use all cpus available
    prefetch_factor = 4

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        pin_memory=True  
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        pin_memory=True
    )


    # criterion = nn.MSELoss()
    model_config = get_model(for_training=True, load_weights=False)
    # criterion = nn.SmoothL1Loss()
    criterion = HybridLoss(img_width=640, img_height=360, margin_px=15)
    optimizer = torch.optim.Adam(model_config.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training loop
    train_losses, val_losses = [], []

    if args.check_error:
        print("Calculating pixel errors in the dataset…")
        model_config.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        compute_pixel_errors(model_config, dataset, DEVICE)
        return

    for epoch in range(NUM_EPOCHS):
        model_config.train()
        running_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model_config(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model_config.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model_config(images)
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
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.savefig("Model/loss_plot.png")

    accepted_tolerance=compute_pixel_errors(model_config, dataset, DEVICE,args.see_points)
    try:
        with open("Model/accepted_tolerance.txt", "r") as f:
            old_tolerance = float(f.read().strip())
    except FileNotFoundError:
        old_tolerance = 0.0
    if accepted_tolerance > old_tolerance:
        print("The model has improved its accuracy, saving new model.")
        os.makedirs("Model", exist_ok=True)
        torch.save(model_config.state_dict(), MODEL_PATH)
        with open("Model/accepted_tolerance.txt", "w") as f:
            f.write(f"{accepted_tolerance:.2f}")
    else:
        print("The model has not improved its accuracy, new model will not be saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for frame extraction and manual ball annotation"
    )

    parser.add_argument(
        "--check_error",
        action="store_true",
        help="Calculates the mean error of the model’s predictions in pixels",
    )

    parser.add_argument(
        "--see_points",
        action="store_true",
        help="Visualizes the predicted and ground-truth coordinates on a plot",
    )


    args = parser.parse_args()
    main(args)
