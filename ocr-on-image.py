import cv2
import os
import pytesseract
import csv
from datetime import datetime
from ultralytics import YOLO

# URL of the IP camera stream
stream_url = "http://192.168.29.207:4747/video"

# Specify the directory where the image will be saved
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Filename for the captured image
image_filename = "captured_image.png"
image_path = os.path.join(save_dir, image_filename)

# CSV file path
csv_filename = 'ocr_output.csv'
csv_path = os.path.join(save_dir, csv_filename)

# Open a connection to the IP camera stream
cap = cv2.VideoCapture(stream_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

# Read a frame from the stream
ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit(1)

# Save the captured frame as a PNG image file
cv2.imwrite(image_path, frame)
print(f"Image captured and saved as '{image_path}'")

# Load the YOLOv8 model (assuming yolov8n.pt is a trained model for license plate detection)
model = YOLO('yolov8n.pt')

# Define a placeholder for class names, update it with actual class names if available
class_names = ['background', 'license_plate']  # Placeholder list

# Perform object detection on the saved image file
results = model(image_path)

# Process the results
detection_data = results[0]  # Access the first (and only) batch of results
for detection in detection_data:
    # Extract bounding boxes, class IDs, and confidence scores
    x1, y1, x2, y2 = detection[:4]  # Bounding box coordinates
    confidence = detection[4]       # Confidence score
    class_id = int(detection[5])    # Class ID

    # Crop the detected license plate area
    cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]

    # Convert cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing for better OCR performance
    preprocessed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Save the preprocessed cropped image
    preprocessed_image_filename = f'preprocessed_cropped_{image_filename}'
    preprocessed_image_path = os.path.join(save_dir, preprocessed_image_filename)
    cv2.imwrite(preprocessed_image_path, preprocessed_image)
    print(f"Preprocessed cropped image saved as '{preprocessed_image_path}'")

    # Perform OCR using Tesseract
    try:
        text = pytesseract.image_to_string(preprocessed_image)
        if not text.strip():  # Check if the OCR result is empty
            raise ValueError("OCR did not detect any text.")
    except Exception as e:
        print(f"Error: OCR failed with exception: {e}")
        cap.release()
        cv2.destroyAllWindows()
        exit(1)

    # Print the OCR output to the terminal
    print(f"OCR Output: {text.strip()}")

    # Save the OCR output to the CSV file with a timestamp
    try:
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Detected Text'])  # Write header if file is new
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text.strip()])
        print(f"OCR output appended to '{csv_path}'")
    except Exception as e:
        print(f"Error: Failed to write to CSV with exception: {e}")
        cap.release()
        cv2.destroyAllWindows()
        exit(1)

    # Exit the program after OCR is completed
    cap.release()
    cv2.destroyAllWindows()
    exit(0)
