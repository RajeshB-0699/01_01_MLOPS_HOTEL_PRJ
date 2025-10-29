# import logging
# import os
# from datetime import datetime

# LOGS_DIR = 'logs'
# os.makedirs(LOGS_DIR, exist_ok=True)

# LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")



# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
# )

# def get_log(name):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     return logger
import os
import logging
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
)

def get_log(name):
    return logging.getLogger(name)
