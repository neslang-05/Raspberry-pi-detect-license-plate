import cv2
import torch
import hydra
import pytesseract
from PIL import Image, ImageFilter
from pathlib import Path
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import csv

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    # Use Raspberry Pi camera as source
    cfg.source = "http://192.168.29.207:4747/video"  # 0 represents the default camera on Raspberry Pi

    # Initialize the predictor
    predictor = DetectionPredictor(cfg)
    predictor()

    # Capture video stream from Raspberry Pi camera
    cap = cv2.VideoCapture(cfg.source)
    if not cap.isOpened():
        print(f"Failed to open camera: {cfg.source}")
        return

    # Create a CSV file to store the results
    csv_file = open('license_plates.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'License Plate'])

    while True:
        # Read frame from the stream
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Perform object detection
        results = predictor(frame)

        # Process detected license plates
        for result in results:
            for box in result.boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy.tolist()

                # Check if the box is a license plate
                if box.cls == 0:  # Assuming 0 is the class index for license plates
                    # Extract the license plate region
                    license_plate_img = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Save the captured license plate image
                    plates_folder = Path(ROOT) / "plates"
                    plates_folder.mkdir(parents=True, exist_ok=True)
                    plate_image_path = plates_folder / "captured_plate.png"
                    cv2.imwrite(str(plate_image_path), license_plate_img)

                    # Preprocess the image for OCR
                    preprocessed_image_path = preprocess_image(plate_image_path)

                    # Perform OCR
                    text = perform_ocr(preprocessed_image_path)

                    # Write the recognized text to the CSV file
                    timestamp = cv2.getTickCount() / cv2.getTickFrequency()  # Get current timestamp
                    csv_writer.writerow([timestamp, text.strip()])
                    print(f"Recognized License Plate: {text.strip()}")

                    # Display the frame with the detected license plate (optional)
                    # cv2.imshow("Raspberry Pi Camera", frame)  # Display the frame with OCR results

                    # Exit after processing the first license plate
                    # cap.release()
                    # cv2.destroyAllWindows()
                    # return

        # Display the frame with OCR results (optional)
        cv2.imshow("Raspberry Pi Camera", frame)  # Display the frame with OCR results

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

def preprocess_image(image_path):
    """Performs image preprocessing for better OCR accuracy."""
    image = Image.open(image_path).convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    preprocessed_image_path = Path(image_path).parent / "preprocessed_plate.png"
    image.save(preprocessed_image_path)
    return preprocessed_image_path

def perform_ocr(image_path):
    """Performs OCR on the specified image."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 6')
    return text

if __name__ == "__main__":
    predict()