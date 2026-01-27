#!/usr/bin/env python3
import subprocess
import signal
import sys
import os
from auto_utils.logger import get_logger
from Config.env_config import *


# List to keep track of subprocesses
processes = []

log= get_logger("MAIN")

def start():
    if os.path.isfile(CONTROLMPP_BIN):
        log.info("Starting ControlMPP...")
        control_proc = subprocess.Popen([CONTROLMPP_BIN], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(control_proc)
    else:
        log.error(f"ControlMPP binary not found at {CONTROLMPP_BIN}. Please compile it first.")
        stop()

    if os.path.isfile(VIDEOCAPTURE_SCRIPT) and os.path.isfile(MODEL_PATH):
        log.info("Starting VideoCapture...")
        video_proc = subprocess.Popen(
            ["python3", VIDEOCAPTURE_SCRIPT, "--source", "camera_usb"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(video_proc)
    else:
        log.error(f"VideoCapture script or model not found. Please check paths:\n"
                  f"VideoCapture: {VIDEOCAPTURE_SCRIPT}\nModel: {MODEL_PATH}")
        stop()

def stop(signum=None, frame=None):
    log.info("\nStopping AutoBall...")
    for proc in processes:
        if proc.poll() is None:  
            proc.terminate()    
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.info(f"Killing PID {proc.pid}")
                proc.kill()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    start()
    log.info("AutoBall running. Press Ctrl+C to stop.")
    try:
        for proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        stop()
