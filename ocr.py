import sys
import pytesseract
import cv2
import os
import csv
from datetime import datetime
from collections import Counter

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Enhance contrast
    enhanced_image = cv2.equalizeHist(blurred_image)
    
    # Use adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    
    # Optionally resize the image
    height, width = morph_image.shape
    new_width = 800
    new_height = int((new_width / width) * height)
    resized_image = cv2.resize(morph_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_image

def perform_ocr(image_path, attempts=10):
    ocr_results = []
    for _ in range(attempts):
        try:
            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)
            
            # Perform OCR using Tesseract with configuration parameters
            custom_config = r'--oem 3 --psm 8'
            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
            
            if text.strip():  # Check if the OCR result is not empty
                ocr_results.append(text.strip())
        except Exception as e:
            print(f"Error: OCR failed with exception: {e}")

    if not ocr_results:
        raise ValueError("OCR did not detect any text after multiple attempts.")

    # Find the most common result
    most_common_text = Counter(ocr_results).most_common(1)[0][0]
    return most_common_text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: File '{image_path}' does not exist.")
        sys.exit(1)

    try:
        result_text = perform_ocr(image_path)
        
        # Print the OCR output to the terminal
        print(f"OCR Output: {result_text}")

        # Save the OCR output to the CSV file with a timestamp
        save_dir = "captured_images"
        csv_filename = 'ocr_output.csv'
        csv_path = os.path.join(save_dir, csv_filename)

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Detected Text'])  # Write header if file is new
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), result_text])
        print(f"OCR output appended to '{csv_path}'")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
