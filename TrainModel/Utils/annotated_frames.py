# Recargamos las librerías y ejecutamos nuevamente el script de anotación
import cv2
import os
import csv
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import time
from Config.model_config import transform_config, get_model

SCALE_FACTOR = 2.0  # o 2.0 según tu pantalla
MODEL_PATH = "Model/Autoball_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STADISTICS_CSV = "Utils/Stadistics.csv"

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
    init_time = time.time()
    n_annotated_images= 0
    output_dir.mkdir(parents=True, exist_ok=True)
    label_csv_path = Path(label_csv_path)
    label_csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in input_dir.glob("*.jpg")])

    mode = 'a' if continue_annotation else 'w'
    try:

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
                    trained_model = get_model(for_training=False, load_weights=True, weights_path=MODEL_PATH)
            for img_number,img_path in enumerate(image_files[num_backup:],start=num_backup):

                print(f"Anotando imagen {img_number + 1}/{len(image_files)}: {img_path.name}")

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Error reading {img_path}")
                    continue

                if assisted:
                    trained_model = get_model(for_training=False, load_weights=True, weights_path=MODEL_PATH)
                    x_pred,y_pred=model_assistant(img_path, trained_model, transform_config)    
                    x_disp_pred = int(x_pred * SCALE_FACTOR)
                    y_disp_pred = int(y_pred * SCALE_FACTOR)
                    print(f"Predicción asistida: x={x_pred}, y={y_pred} (pantalla: {x_disp_pred}, {y_disp_pred})")
                
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
                            n_annotated_images += 1
                            annotated_img = clone.copy()
                            cv2.circle(annotated_img, (clicked[0], clicked[1]), circle_radius, (0, 0, 255), 2)
                            # save_path = output_dir / img_path.name
                            # cv2.imwrite(str(save_path), annotated_img)
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
                        end_time = time.time()
                        elapsed_time = end_time - init_time
                        # Calculo de la tasa de anotación
                        if n_annotated_images > 0:
                            annotation_rate = n_annotated_images / elapsed_time
                            # Guardar estadísticas en CSV
                            try: 
                                with open(STADISTICS_CSV, 'a', newline='') as stats_file:
                                    stats_writer = csv.writer(stats_file)
                                    stats_writer.writerow([n_annotated_images, elapsed_time, annotation_rate])
                            except Exception as e:
                                print(f"Error al guardar estadísticas: {e}")
                        try: 
                            with open(os.path.join(output_dir, 'num_backup.txt'), 'w') as f:
                                f.write(f"{img_number}\n")
                        except Exception as e:
                            print(f"Error al guardar el número de backup: {e}")
                        print("Salida anticipada.")
                        cv2.destroyAllWindows()
                        return
    
    except Exception as e:
        print(f"Error durante la anotación: {e}")
                    



