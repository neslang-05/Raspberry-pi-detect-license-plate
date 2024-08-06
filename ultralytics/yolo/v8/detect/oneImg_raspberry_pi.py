# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import cv2
import torch
import pytesseract
import numpy as np
from PIL import Image, ImageFilter
import hydra
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics import YOLO  # Correct import for the YOLO class

class DetectionPredictor:
    def __init__(self, args):
        self.model = YOLO(args.model)  # Load the YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if args.half:
            self.model.half()
        self.args = args

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.args.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.args.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def detect_and_read_plate(self, img):
        img_preprocessed = self.preprocess(img)
        img_preprocessed = img_preprocessed.unsqueeze(0)  # Add batch dimension

        preds = self.model(img_preprocessed)

        preds = self.postprocess(preds, img_preprocessed, img)

        if not os.path.exists('plates'):
            os.makedirs('plates')

        for det in preds:
            for *xyxy, conf, cls in det:
                if cls == self.model.names.index('license_plate'):  # Assuming class name is 'license_plate'
                    x1, y1, x2, y2 = map(int, xyxy)
                    plate_img = img[y1:y2, x1:x2]
                    plate_image_path = os.path.join('plates', "plate.png")
                    cv2.imwrite(plate_image_path, plate_img)

                    preprocessed_image_path = preprocess_image(plate_image_path)

                    image = Image.open(preprocessed_image_path)
                    text = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 6')

                    print(f"Recognized License Plate: {text.strip()}")
                    return

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    preprocessed_image_path = os.path.join('plates', "preprocessed_plate.png")
    image.save(preprocessed_image_path)
    return preprocessed_image_path

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)

    cap = cv2.VideoCapture("http://192.168.29.207:4747/video")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture image")
        return

    predictor.detect_and_read_plate(frame)

if __name__ == "__main__":
    predict()
