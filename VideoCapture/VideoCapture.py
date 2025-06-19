from VideoCapture_lib import USBCameraCapture, VideoFileCapture
import argparse
import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50

MODEL_PATH = "../TrainModel/Model/Autoball_model.pth"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    video_path="../TrainModel/InputVideos/test3.mp4"
    parser = argparse.ArgumentParser(description="Captura de vídeo desde cámara USB o archivo de vídeo.")
    parser.add_argument("--source", type=str, default="file", help="Origen de la captura ('camera' o 'file')")
    parser.add_argument("--framerate", type=int, default=30, help="Tasa de fotogramas por segundo (default: 30)")
    parser.add_argument("--video_file", type=str, default=video_path, help="Ruta al archivo de vídeo (opcional)")
    parser.add_argument("--images_per_sec", type=int, default=30, help="Imagenes por segundo enviadas al modelo (default: 10)")
    args = parser.parse_args()
    frame_duration = 1.0 / args.framerate  # segundos por frame
    if args.video_file:
        capture = VideoFileCapture(args.video_file)
    else:
        capture = USBCameraCapture(camera_index=args.camera, framerate=args.framerate)

    if capture.start():
        frame_interval = 1 / args.images_per_sec  # segundos
        try:
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval().to(DEVICE)

            transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            capture.stop()
            cv2.destroyAllWindows()

        
        try:
            last_time = time.time()
            while True:
                current_time = time.time()
                if current_time - last_time >= frame_interval:
                    frame = capture.read()
                    if frame is not None:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        input_tensor = transform(img).unsqueeze(0).to(DEVICE)  
                        with torch.no_grad():
                            output = model(input_tensor).squeeze().cpu().numpy()  
                        x_pred, y_pred = float(output[0]), float(output[1])
                        height, width = frame.shape[:2]

                        # Convertir coordenadas normalizadas a píxeles
                        x_pixel = int(x_pred * width)
                        y_pixel = int(y_pred * height)

                        # Dibujar un círculo rojo (radio 8 px, grosor -1 = relleno)
                        cv2.circle(frame, (x_pixel, y_pixel), 8, (0, 0, 255), -1)

                        # Mostrar el frame con la predicción
                        cv2.imshow("Frame", frame)
                        last_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                
        finally:
            capture.stop()
            cv2.destroyAllWindows()
    
    