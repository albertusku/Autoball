import os
import torch

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLMPP_BIN = os.path.join(PROJECT_DIR, "ControlMPP/bin/ControlMPP")
VIDEOCAPTURE_SCRIPT = os.path.join(PROJECT_DIR, "VideoCapture/VideoCapture.py")
MODEL_PATH = os.path.join(PROJECT_DIR,"TrainModel/Model/Autoball_model.pth")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
VIDEO_PATH = os.path.join(PROJECT_DIR, "TrainModel/InputVideos")
SOCK_PATH = "/tmp/pid_socket"
RSTP_URL = "@192.168.1.57:8554/live"
RTSP_NAME= "admin"
RTSP_PASS= "user"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-4
N_SAMPLES = 50