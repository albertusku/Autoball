import cv2
import pandas as pd
from pathlib import Path

def revisar_labels(csv_path, images_dir, circle_radius=10):
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    if not csv_path.exists():
        print(f" CSV no encontrado: {csv_path}")
        return
    if not images_dir.exists():
        print(f"Carpeta de imágenes no encontrada: {images_dir}")
        return

    data = pd.read_csv(csv_path)

    for i, row in data.iterrows():
        img_path = images_dir / row['image']
        if not img_path.exists():
            print(f"[AVISO] Imagen no encontrada: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] No se pudo leer: {img_path}")
            continue

        x, y = int(row['x']), int(row['y'])
        img_disp = img.copy()
        cv2.circle(img_disp, (x, y), radius=circle_radius, color=(0, 0, 255), thickness=2)

        cv2.namedWindow("Verificar etiqueta", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Verificar etiqueta", 100, 100)
        cv2.imshow("Verificar etiqueta", img_disp)

        print(f"[{i+1}/{len(data)}] {row['image']} - Coordenadas: ({x}, {y})")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Salida manual.")
            break
        elif key == ord('n'):
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    revisar_labels(
        csv_path="Labels/test2/labels.csv",
        images_dir="ExtractedFrames/test2"
    )
