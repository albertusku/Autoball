# utils/logger.py
import logging
from datetime import datetime
import os
from Config.env_config import *

def get_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, datetime.now().strftime("autoball_%Y-%m-%d.log"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Evita duplicados
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # También por consola si quieres
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
