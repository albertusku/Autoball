import cv2
import threading
import time
import socket
import time
import errno
import numpy as np
from auto_utils.logger import get_logger
import os
from Config.env_config import *

MAX_SUN_PATH = 104
log= get_logger("VideoCapture")


class BaseCapture:

    def start(self):
        """Starts the capture (if applicable)."""
        raise NotImplementedError
    
    def read(self):
        """Returns the last captured frame as an image (numpy array)."""
        raise NotImplementedError

    def stop(self):
        """Stops the capture and releases resources."""
        raise NotImplementedError
    
    def get_distance_to_middle(self,frame,x,y):
        """
        Calculates the distance from a point (x, y) to the center of the frame.
        
        Args:
            frame (numpy.ndarray): The image frame.
            x (int): x coordinate of the point.
            y (int): y coordinate of the point.
        
        Returns:
            float: Distance to the center of the frame.
        """
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        left_side=False
        if x < 320:
            left_side=True
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        if left_side:
            distance = -distance  # Negate the distance if it is to the left of the center
        self.send_data_to_PID(distance)
        return distance
    
    def send_data_to_PID(self, data: float,
                     socket_path: str = SOCK_PATH,
                     retries: int = 5,
                     backoff_s: float = 0.05) -> bool:
        """
        Sends 'data' as text (e.g., '12.34') via UNIX DGRAM socket to 'socket_path'.
        Returns True if sent; False if retries are exhausted.
        """
        # Quick validations
        if not socket_path or len(socket_path) > MAX_SUN_PATH:
            # Avoid paths that are too long which silently fail in AF_UNIX
            return False

        payload = f"{float(data):.2f}".encode()
        # log.debug(f"Sending data to PID: {payload} to {socket_path}")

        delay = backoff_s
        for _ in range(max(1, retries)):
            try:
                # DGRAM: no connect(); use sendto()
                with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
                    s.sendto(payload, socket_path)
                return True

            except FileNotFoundError:
                # The receiver has not bound yet; wait and retry
                time.sleep(delay)
                delay *= 1.5
            except PermissionError:
                # Insufficient socket permissions
                return False
            except OSError as e:
                # ENOENT: path does not exist yet; EINVAL sometimes if the path is not a socket
                if e.errno in (errno.ENOENT, errno.EINVAL):
                    time.sleep(delay)
                    delay *= 1.5
                    continue
                # Other errors: return False (or re-raise if preferred)
                return False

        return False


        
    

class USBCameraCapture(BaseCapture):
    def __init__(self, camera_index=0, resolution=(640, 360), framerate=30):
        self.rtsp_url = f"rtsp://{RTSP_NAME}:{RTSP_PASS}{RSTP_URL}"
        self.camera_index = camera_index
        self.resolution = resolution
        self.framerate = framerate
        self.running = False
        self.frame = None
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, framerate)
        self.lock = threading.Lock()
        self._thread = None

    def start(self, wait_first_frame=True, first_frame_timeout=2.0):

        if self.cap is None or not self.cap.isOpened():
            log.warning(f"Reopening camera with resolution {self.resolution} and framerate {self.framerate}")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS,          self.framerate)

        if not self.cap.isOpened():
            log.error(f"Failed to open camera. Check if the camera is connected and available."
                        f"url: {self.rtsp_url}")
            return False
        
        self.running = True
        self._thread=threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        if wait_first_frame:
            log.info(f"Waiting for the first frame from camera ...")
            t0 = time.time()
            while self.frame is None and (time.time() - t0) < first_frame_timeout:
                time.sleep(0.01)
            return self.frame is not None

        return True

    def _update(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                log.warning(f"Camera {self.camera_index} not opened, trying to reopen...")
                time.sleep(0.1)
                self.cap = cv2.VideoCapture(self.rtsp_url)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS,          self.framerate)
                continue
            ret, frame = self.cap.read()
            if ret:
                # filename = os.path.join("/home/ruiz17/Autoball/test", f"frame.jpg")
                # log.debug(f"Saving frame in  {filename}")
                # cv2.imwrite(filename, frame)
                with self.lock:
                    self.frame = frame
            else:
                log.error("Failed to capture frame from camera.")
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        log.info("Stopping camera capture...")
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class VideoFileCapture(BaseCapture):
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Cant open video in: {video_path}")

    def start(self):
        if self.cap is None or not self.cap.isOpened():

            self.cap = cv2.VideoCapture(self.video_path)

        if self.cap.isOpened():

            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, 70000)
        else:
            return False

    def read(self):
        """Returns the next frame or None if the video has ended."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        """Releases the video file resource."""
        self.cap.release()
    
