# Recargamos las librerías y ejecutamos nuevamente el script de anotación
import cv2
import os
import csv
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

SCALE_FACTOR = 2.0  # o 2.0 según tu pantalla
MODEL_PATH = "Model/Autoball_model.pth"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_assistant(img_path, model, transform,):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(input_tensor).squeeze().cpu().numpy()
        x_pred = int(pred[0] * orig_w)
        y_pred = int(pred[1] * orig_h)
    
    return x_pred, y_pred
    

def annotate_frames(input_dir, output_dir, label_csv_path, circle_radius=15, continue_annotation=False,assisted=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_csv_path = Path(label_csv_path)
    label_csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in input_dir.glob("*.jpg")])

    mode = 'a' if continue_annotation else 'w'
    with open(label_csv_path, mode=mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':
            writer.writerow(['image', 'x', 'y'])  # Escribe cabecera solo si es nuevo
        num_backup = 1
        if continue_annotation:
            with open(os.path.join(output_dir, 'num_backup.txt'), 'r') as f:
                num_backup = f.readlines()
                num_backup= int(num_backup[0])

        if assisted:
            # Cargar el modelo ResNet50
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval().to(DEVICE)

            # Transformación de imagen
            transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])

        for img_number,img_path in enumerate(image_files[num_backup:],start=num_backup):
            print(f"Anotando imagen {img_number + 1}/{len(image_files)}: {img_path.name}")

            if assisted:
                x_pred,y_pred=model_assistant(img_path, model, transform)
                x_disp_pred = int(x_pred * SCALE_FACTOR)
                y_disp_pred = int(y_pred * SCALE_FACTOR)
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error reading {img_path}")
                continue

            resized_img = cv2.resize(img, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            clone = resized_img.copy()
            clicked = []

            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked.clear()
                    clicked.extend([x, y])
                    temp_img = clone.copy()
                    cv2.circle(temp_img, (x, y), circle_radius, (0, 0, 255), 2)
                    cv2.imshow("Annotate", temp_img)

            if assisted:
                cv2.circle(resized_img, (x_disp_pred, y_disp_pred), 10, (0, 255, 0), 2)
            cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Annotate", 100, 100)
            cv2.setMouseCallback("Annotate", click_event)

            # Mostrar imagen original al principio
            cv2.imshow("Annotate", resized_img)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s'):
                    if clicked:
                        annotated_img = clone.copy()
                        cv2.circle(annotated_img, (clicked[0], clicked[1]), circle_radius, (0, 0, 255), 2)
                        save_path = output_dir / img_path.name
                        cv2.imwrite(str(save_path), annotated_img)
                        writer.writerow([img_path.name, clicked[0]/SCALE_FACTOR, clicked[1]/SCALE_FACTOR])
                        print(f"[GUARDADA] {img_path.name} ,x: {clicked[0]/SCALE_FACTOR}, y: {clicked[1]/SCALE_FACTOR}")
                        break
                    elif assisted:
                        writer.writerow([img_path.name, x_pred, y_pred])
                        print(f"[AUTOMÁTICA] {img_path.name} ,x: {x_pred}, y: {y_pred}")
                        break
                elif key == ord('d'):
                    print(f"[IGNORADA] {img_path.name}")
                    break
                elif key == ord('q'):
                    with open(os.path.join(output_dir, 'num_backup.txt'), 'w') as f:
                        f.write(f"{img_number}\n")
                    print("Salida anticipada.")
                    cv2.destroyAllWindows()
                    return
                    



