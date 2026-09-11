from pathlib import Path
from src.model_service.inference_service import InferenceService

image_path = Path(input("Enter image path: ").strip().strip('"'))
if not image_path.is_file():
    raise FileNotFoundError(image_path)

service = InferenceService()
print(service.process_image(image_path.read_bytes()))

