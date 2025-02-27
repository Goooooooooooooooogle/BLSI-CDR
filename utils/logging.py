import os


import logging
import time
logger = logging.getLogger("BLSICDR_PYTORCH")
log_level = "DEBUG"

log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"logs/log-{}.txt".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))

logger.setLevel(level = log_level)
formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
 
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
 
logger.addHandler(handler)
logger.addHandler(console)
