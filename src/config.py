# src/config.py
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

<<<<<<< HEAD
TOMTOM_API_KEY = os.getenv("TOMTOM_TRAFFIC_API_KEY", None)
=======
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", None)
>>>>>>> 4e99cfca607266ec4b42201e31e44950e52d5ece
THU_DUC_BOUNDS = {
    'north': 10.8700, 'south': 10.7300, 'east': 106.8200, 'west': 106.6800
}