#!/usr/bin/env python3
import subprocess
import signal
import sys
import os
from typing import List

from auto_utils import ManagedProcess, get_logger, launch_process

# Rutas absolutas de los ejecutables
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLMPP_BIN = os.path.join(PROJECT_DIR, "ControlMPP/bin/ControlMPP")
VIDEOCAPTURE_SCRIPT = os.path.join(PROJECT_DIR, "VideoCapture/VideoCapture.py")

# Lista para guardar procesos lanzados
processes: List[ManagedProcess] = []

log = get_logger("MAIN")

def start():
    log.info("Starting ControlMPP...")
    control_proc = launch_process([CONTROLMPP_BIN], logger=log, name="ControlMPP")
    processes.append(control_proc)

    log.info("Starting VideoCapture...")
    video_proc = launch_process(
        [sys.executable, VIDEOCAPTURE_SCRIPT, "--source", "camera"],
        logger=log,
        name="VideoCapture",
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
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning(f"Timeout esperando la terminación del proceso {proc.pid}")
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
