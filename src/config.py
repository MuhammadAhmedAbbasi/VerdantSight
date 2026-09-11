import os


API_KEY = os.getenv("PLANT_API_KEY", "change-this-key")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", "3"))
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.25"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "leaf-pothos-v1")
