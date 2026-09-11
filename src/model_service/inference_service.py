from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO
from src.config import MIN_DETECTION_CONFIDENCE

ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = ROOT / "models" / "leaf_detector.pt"
CLASSIFIER_PATH = ROOT / "models" / "pothos_classifier.pt"
CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]

def block(input_channels, output_channels, pool=False):
    layers = [nn.Conv2d(input_channels, output_channels, 3, padding=1),
              nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self):
        if not DETECTOR_PATH.is_file() or not CLASSIFIER_PATH.is_file():
            raise FileNotFoundError("Model files are missing from src/models")
        super().__init__()
        self.conv1 = block(3, 64)
        self.conv2 = block(64, 128, True)
        self.res1 = nn.Sequential(block(128, 128), block(128, 128))
        self.conv3 = block(128, 256, True)
        self.conv4 = block(256, 512, True)
        self.res2 = nn.Sequential(block(512, 512), block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, 3))

    def forward(self, image):
        output = self.conv2(self.conv1(image))
        output = self.res1(output) + output
        output = self.conv4(self.conv3(output))
        output = self.res2(output) + output
        return self.classifier(output)

def make_square(image):
    side = max(image.width, image.height)
    left = (side - image.width) // 2
    top = (side - image.height) // 2
    border = (left, top, side-image.width-left, side-image.height-top)
    return ImageOps.expand(image, border=border, fill=(0, 0, 0))

class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(DETECTOR_PATH))
        self.classifier = ResNet9().to(self.device)
        self.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=self.device))
        self.classifier.eval()
        self.transform = transforms.Compose([
            transforms.Lambda(make_square),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def process_image(self, image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Received data is not a valid image")
        result = self.detector.predict(image, conf=MIN_DETECTION_CONFIDENCE, verbose=False)[0]
        detections = []
        with torch.inference_mode():
            for x1, y1, x2, y2 in result.boxes.xyxy.int().tolist():
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
                probabilities = torch.softmax(self.classifier(tensor), dim=1)[0]
                confidence, class_id = probabilities.max(0)
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": CLASS_NAMES[class_id.item()],
                    "confidence": round(confidence.item(), 4),
                })
        return {
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "detections": detections,
        }
