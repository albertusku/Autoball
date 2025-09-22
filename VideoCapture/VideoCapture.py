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
import subprocess
from flask import Flask, Response
import os

MODEL_PATH = "TrainModel/Model/Autoball_model.pth"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log= get_logger("VideoCapture")
RTSP_URL = "rtsp://localhost:8554/mystream"


app = Flask(__name__)

def process_frame(frame, model_config, transform_config, DEVICE, width, height):
    """Procesa un frame: predicción y dibujo del balón"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform_config(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model_config(input_tensor).squeeze().cpu().numpy()
    x_pred, y_pred = float(output[0]), float(output[1])

    # Convertir coordenadas normalizadas a píxeles
    x_pixel = int(x_pred * width)
    y_pixel = int(y_pred * height)
    distance = capture.get_distance_to_middle(frame, x_pixel, y_pixel)
    # Dibujar un círculo rojo
    cv2.circle(frame, (x_pixel, y_pixel), 8, (0, 0, 255), -1)
    # filename = os.path.join("/home/ruiz17/Autoball/test", f"frame.jpg")
    # cv2.imwrite(filename, frame)
    return frame

def generate_frames(capture, model_config, transform_config, DEVICE, width, height, frame_duration):
    """Generador de frames para Flask (stream MJPEG)"""
    last_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_time >= frame_duration:
            frame = capture.read()
            if frame is not None:
                frame = process_frame(frame, model_config, transform_config, DEVICE, width, height)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_time = time.time()

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

    log.info(f"Starting video capture with source in: {args.source}, framerate: {args.framerate},"
              f"and frame_duration: {frame_duration}")

    if args.source == "file":
        capture = VideoFileCapture(args.video_file)
    elif args.source == "camera":
        capture = USBCameraCapture(framerate=args.framerate)
    
    log.info(f"Getting model from: {MODEL_PATH}")
    try:
        model_config = get_model(for_training=False, load_weights=True, weights_path=MODEL_PATH)
    except Exception as e:
        log.error(f"Error loading model: {e}")
        raise e
    log.info("Starting video capture...")
    if capture.start():
        try:
            first_frame = capture.read()
            if first_frame is None:
                raise RuntimeError("Not able to find the first frame to init RTSP.")
            height, width = first_frame.shape[:2]
            if args.test:
                from flask import Flask, Response
                app = Flask(__name__)
                @app.route('/video_feed')
                def video_feed():
                    return Response(
                        generate_frames(capture,model_config,transform_config, DEVICE, width, height, frame_duration),
                        mimetype='multipart/x-mixed-replace; boundary=frame'
                    )
                app.run(host='0.0.0.0', port=5000, debug=False)
            else:
                log.info(f"Starting frame generation (no Flask)")
                last_time = time.time()
                while True:
                    current_time = time.time()
                    if current_time - last_time >= frame_duration:
                        frame = capture.read()
                        if frame is not None:
                            frame = process_frame(frame, model_config, transform_config, DEVICE, width, height)
                            if args.source == "file":
                                cv2.imshow("Frame", frame)
                            if cv2.waitKey(1) & 0xFF == ord('q') and args.source == "file":
                                break
                        last_time = time.time()
                

        finally:
            capture.stop()
            cv2.destroyAllWindows()
    
    
