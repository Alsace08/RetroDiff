import os
import json
import logging
from pathlib import Path
from datetime import datetime


def get_current_datetime():
    return datetime.now().strftime('%Y_%m_%d_%H:%M:%S')


def get_logger(name, current_time):
    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(filename=f"./experiments/logging/{current_time}.log",
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        force=True)

    logger = logging.getLogger(name)

    return logger
    
