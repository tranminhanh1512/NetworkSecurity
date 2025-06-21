import logging
import os
from datetime import datetime
"""
This module is used to create a logger object to systematically track events and debug information
"""
# Generate a unique log filename based on the current date and time
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory to store logs
logs_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(logs_dir, exist_ok=True)

# Final log file path to be used in the logger configuration
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure the logging settings:
# - Write logs to the generated log file
# - Use a format that includes timestamp, line number, logger name, log level, and the log message
# - Set the log level to INFO (only INFO and more severe messages will be recorded)
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = '[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level = logging.INFO
)

logger = logging.getLogger('NetworkSecurityLogger')