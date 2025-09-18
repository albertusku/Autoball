#!/usr/bin/env python3
import subprocess
import signal
import sys
import os
from auto_utils.logger import get_logger

# Rutas absolutas de los ejecutables
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLMPP_BIN = os.path.join(PROJECT_DIR, "ControlMPP/bin/ControlMPP")
VIDEOCAPTURE_SCRIPT = os.path.join(PROJECT_DIR, "VideoCapture/VideoCapture.py")

# Lista para guardar procesos lanzados
processes = []

log= get_logger("MAIN")

def start():
    log.info("Starting ControlMPP...")
    control_proc = subprocess.Popen([CONTROLMPP_BIN], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append(control_proc)

    log.info("Starting VideoCapture...")
    video_proc = subprocess.Popen(
        ["python3", VIDEOCAPTURE_SCRIPT, "--source", "camera"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    processes.append(video_proc)

def stop(signum=None, frame=None):
    log.info("\nStopping AutoBall...")
    for proc in processes:
        if proc.poll() is None:  # sigue vivo
            proc.terminate()     # señal SIGTERM
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.info(f"Forzando kill a PID {proc.pid}")
                proc.kill()
    sys.exit(0)

if __name__ == "__main__":
    # Capturar Ctrl+C
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    start()
    log.info("AutoBall running. Press Ctrl+C to stop.")

    # Mantener vivo hasta que los hijos terminen
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        stop()
