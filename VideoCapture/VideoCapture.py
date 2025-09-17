from VideoCapture_lib import USBCameraCapture, VideoFileCapture
import argparse
import cv2
import time
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
from Config.model_config import transform_config, get_model
from auto_utils.logger import get_logger

MODEL_PATH = "../TrainModel/Model/Autoball_model.pth"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log= get_logger("VideoCapture")
RTSP_URL = "rtsp://localhost:8554/mystream"


if __name__ == "__main__":
    video_path="../TrainModel/InputVideos/test5.mp4"
    parser = argparse.ArgumentParser(description="Captura de vídeo desde cámara USB o archivo de vídeo.")
    parser.add_argument("--source", type=str, default="file", help="Origen de la captura ('camera' o 'file')")
    parser.add_argument("--framerate", type=int, default=30, help="Tasa de fotogramas por segundo (default: 30)")
    parser.add_argument("--video_file", type=str, default=video_path, help="Ruta al archivo de vídeo (opcional)")
    parser.add_argument("--images_per_sec", type=int, default=30, help="Imagenes por segundo enviadas al modelo (default: 10)")
    parser.add_argument("--test",action="store_true",help="True si se ejecuta el test")
    args = parser.parse_args()
    frame_duration = 1.0 / args.framerate  # segundos por frame

    log.info(f"Starting video capture with source in: {args.source}, framerate: {args.framerate}, video_file: {args.video_file}"
              f", images_per_sec: {args.images_per_sec} and frame_duration: {frame_duration}")

    if args.source == "file":
        capture = VideoFileCapture(args.video_file)
    elif args.source == "camera":
        capture = USBCameraCapture(framerate=args.framerate)
    
    log.info(f"Getting model from: {MODEL_PATH}")
    model_config = get_model(for_training=False, load_weights=True, weights_path=MODEL_PATH)
    log.info("Starting video capture...")
    if capture.start():
        try:
            last_time = time.time()
            first_frame = capture.read()
            if first_frame is None:
                raise RuntimeError("No se pudo capturar el primer frame para inicializar RTSP.")
            height, width = first_frame.shape[:2]
            ffmpeg_cmd = [
                "ffmpeg",
                "-re",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",  # Entrada por stdin
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-f", "rtsp",
                RTSP_URL
                ]
            proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            while True:
                current_time = time.time()
                if current_time - last_time >= frame_duration:
                    frame = capture.read()
                    if frame is not None:
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        input_tensor = transform_config(img).unsqueeze(0).to(DEVICE)  
                        with torch.no_grad():
                            output = model_config(input_tensor).squeeze().cpu().numpy()  
                        x_pred, y_pred = float(output[0]), float(output[1])

                        # Convertir coordenadas normalizadas a píxeles
                        x_pixel = int(x_pred * width)
                        y_pixel = int(y_pred * height)
                        distance=capture.get_distance_to_middle(frame, x_pixel, y_pixel)
                        # Dibujar un círculo rojo (radio 8 px, grosor -1 = relleno)
                        cv2.circle(frame, (x_pixel, y_pixel), 8, (0, 0, 255), -1)
                        if args.test and proc is not None:
                            proc.stdin.write(frame.tobytes())
                        if args.source == "file" :
                            cv2.imshow("Frame", frame)
                            
                        last_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q') and args.source == "file":
                    break

                
        finally:
            capture.stop()
            cv2.destroyAllWindows()
            if proc:
                proc.stdin.close()
                proc.wait()
    
    
