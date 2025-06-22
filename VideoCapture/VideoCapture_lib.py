import cv2
import threading
import time

import numpy as np


class BaseCapture:

    def start(self):
        """Inicia la captura (si aplica)."""
        raise NotImplementedError
    
    def read(self):
        """Devuelve el último frame capturado como imagen (numpy array)."""
        raise NotImplementedError

    def stop(self):
        """Detiene la captura y libera recursos."""
        raise NotImplementedError
    
    def get_distance_to_middle(self,frame,x,y):
        """
        Calcula la distancia desde un punto (x, y) al centro del frame.
        
        Args:
            frame (numpy.ndarray): El frame de la imagen.
            x (int): Coordenada x del punto.
            y (int): Coordenada y del punto.
        
        Returns:
            float: Distancia al centro del frame.
        """
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        left_side=False
        if x < 320:
            left_side=True
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if left_side:
            distance = -distance  # Negar la distancia si está a la izquierda del centro
        return distance
    

class USBCameraCapture(BaseCapture):
    def __init__(self, camera_index=0, resolution=(640, 360), framerate=30):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


class VideoFileCapture(BaseCapture):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    def start(self):
        if self.cap is None or not self.cap.isOpened():
            # Intentamos reabrir el vídeo si fue cerrado
            self.cap = cv2.VideoCapture(self.video_path)

        if self.cap.isOpened():
            # Reiniciar al primer frame
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, 70000)
        else:
            return False

    def read(self):
        """Devuelve el siguiente frame o None si terminó el vídeo."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        """Libera el recurso del archivo de vídeo."""
        self.cap.release()
    
