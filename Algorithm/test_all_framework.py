"""Detect leaves with YOLO and classify them as healthy, chlorosis, or necrosis."""

from pathlib import Path
import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Lambda, Resize, ToTensor
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL = ROOT / "Algorithm" / "runs" / "detect" / "health_leaf" / "office_plant_nano" / "weights" / "best.pt"
CLASSIFIER_MODEL = ROOT / "Algorithm" / "runs" / "classify" / "pothos_resnet9" / "best.pt"
CLASS_NAMES = ["healthy", "chlorosis", "necrosis"]
OUTPUT_DIR = ROOT / "Algorithm" / "predictions" / "pothos_inference"
DETECTION_CONFIDENCE = 0.5
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
LABEL_FONT_SCALE = 0.9
LABEL_THICKNESS = 2
BOX_THICKNESS = 3

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, number_of_classes=3):
        super().__init__()
        self.conv1 = conv_block(3, 64); self.conv2 = conv_block(64, 128, True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, True); self.conv4 = conv_block(256, 512, True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, number_of_classes))
    def forward(self, x):
        x = self.conv2(self.conv1(x)); x = self.res1(x) + x
        x = self.conv4(self.conv3(x)); x = self.res2(x) + x
        return self.classifier(x)

def square(image):
    side = max(image.width, image.height); left = (side - image.width) // 2; top = (side - image.height) // 2
    return ImageOps.expand(image, border=(left, top, side-image.width-left, side-image.height-top), fill=(0, 0, 0))

def load_models(device):
    if not YOLO_MODEL.is_file(): raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL}")
    if not CLASSIFIER_MODEL.is_file(): raise FileNotFoundError(f"Classifier model not found: {CLASSIFIER_MODEL}")
    detector = YOLO(str(YOLO_MODEL)); classifier = ResNet9().to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location=device)); classifier.eval()
    return detector, classifier, Compose([Lambda(square), Resize((256, 256)), ToTensor()])

def annotate(frame, detector, classifier, transform, device):
    result = detector.predict(frame, conf=DETECTION_CONFIDENCE, verbose=False)[0]; output = frame.copy()
    with torch.inference_mode():
        for box in result.boxes.xyxy.int().tolist():
            x1, y1, x2, y2 = box; x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            probabilities = torch.softmax(classifier(tensor), dim=1)[0]; confidence, class_id = probabilities.max(0)
            label = f"{CLASS_NAMES[class_id.item()]} {confidence.item() * 100:.1f}%"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICKNESS)
            size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)
            text_y = max(y1, size[1] + 12)
            cv2.rectangle(output, (x1, text_y-size[1]-12),
                          (x1+size[0]+12, text_y), (0, 255, 0), -1)
            cv2.putText(output, label, (x1+6, text_y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE,
                        (0, 0, 0), LABEL_THICKNESS, cv2.LINE_AA)
    return output, len(result.boxes)

def preview(image):
    h, w = image.shape[:2]; scale = min(DISPLAY_WIDTH/w, DISPLAY_HEIGHT/h, 1.0)
    shown = cv2.resize(image, (int(w*scale), int(h*scale))) if scale < 1 else image
    cv2.imshow("Leaf result - press ESC to close", shown)

def main():
    path = Path(input("Enter image or video path: ").strip().strip('"')).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"File not found: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); detector, classifier, transform = load_models(device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(str(path));
        if not cap.isOpened(): raise RuntimeError(f"Could not open video: {path}")
        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_path = OUTPUT_DIR / f"{path.stem}_result.mp4"; writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok: break
            output, _ = annotate(frame, detector, classifier, transform, device); writer.write(output); preview(output)
            if cv2.waitKey(1) & 0xFF == 27: break
        cap.release(); writer.release(); print(f"Saved result: {out_path}")
    else:
        image = cv2.imread(str(path));
        if image is None: raise RuntimeError(f"Could not open image: {path}")
        output, count = annotate(image, detector, classifier, transform, device); out_path = OUTPUT_DIR / f"{path.stem}_result.jpg"; cv2.imwrite(str(out_path), output); preview(output); cv2.waitKey(0)
        print(f"Detected leaves: {count}"); print(f"Saved result: {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
