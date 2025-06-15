# Recargamos las librerías y ejecutamos nuevamente el script de anotación
import cv2
import os
import csv
from pathlib import Path

SCALE_FACTOR = 2  # o 2.0 según tu pantalla

def annotate_frames(input_dir, output_dir, label_csv_path, circle_radius=15, continue_annotation=False):
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

        for img_number,img_path in enumerate(image_files[num_backup:],start=num_backup):
            print(f"Anotando imagen {img_number + 1}/{len(image_files)}: {img_path.name}")
            
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

            cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Annotate", 100, 100)
            cv2.setMouseCallback("Annotate", click_event)

            # Mostrar imagen original al principio
            cv2.imshow("Annotate", img)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('s') and clicked:
                    annotated_img = clone.copy()
                    cv2.circle(annotated_img, (clicked[0], clicked[1]), circle_radius, (0, 0, 255), 2)
                    save_path = output_dir / img_path.name
                    cv2.imwrite(str(save_path), annotated_img)
                    writer.writerow([img_path.name, clicked[0], clicked[1]])
                    print(f"[GUARDADA] {img_path.name} ,x: {clicked[0]}, y: {clicked[1]}")
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
                    



