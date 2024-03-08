import os
import warnings
import logging
import time

def setlogger(logger):
    warnings.filterwarnings('ignore')
    # neglect warning, set logger
    if not os.path.exists(os.path.join(os.getcwd(),'log','log_output')):
        os.makedirs(os.path.join(os.getcwd(),'log','log_output'))
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(os.getcwd(),'log','log_output','log'+time.strftime('%m%d%H%M%S')+'.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger