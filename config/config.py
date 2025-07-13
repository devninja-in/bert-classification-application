import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Application settings
APP_NAME = "BERT Text Classification"
APP_ICON = "🤖"
APP_LAYOUT = "wide"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", logging.INFO)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"

# Model settings
DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_TEST_SIZE = 0.2

# UI settings
PRIMARY_COLOR = "#4B0082"
SECONDARY_COLOR = "#1E90FF"
HIGHLIGHT_BG_COLOR = "#F0F8FF"