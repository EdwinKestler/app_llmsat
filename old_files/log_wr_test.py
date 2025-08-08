import logging
import os
import sys
timestamp = "test"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, f"process_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("Test log")