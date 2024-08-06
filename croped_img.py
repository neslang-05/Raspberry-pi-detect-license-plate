import cv2
import os
from ultralytics import YOLO

# URL of the IP camera stream
stream_url = "http://192.168.29.207:4747/video"

# Specify the directory where the image will be saved
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Filename for the captured image
image_filename = "captured_image.png"
image_path = os.path.join(save_dir, image_filename)

# Open a connection to the IP camera stream
cap = cv2.VideoCapture(stream_url)

# Check if the connection is successful
if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    # Read a frame from the stream
    ret, frame = cap.read()

    if ret:
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

            # Draw bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # If you have a list of class names, use it to label the boxes
            if class_id < len(class_names):
                label = f'{class_names[class_id]}: {confidence:.2f}'
            else:
                label = f'Class {class_id}: {confidence:.2f}'

            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop the detected license plate area
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]

            # Convert cropped image to grayscale
            gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing for better OCR performance
            # You can use techniques such as GaussianBlur, adaptive thresholding, etc.
            preprocessed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Save the preprocessed cropped image
            preprocessed_image_filename = f'preprocessed_cropped_{image_filename}'
            preprocessed_image_path = os.path.join(save_dir, preprocessed_image_filename)
            cv2.imwrite(preprocessed_image_path, preprocessed_image)
            print(f"Preprocessed cropped image saved as '{preprocessed_image_path}'")

            # Optionally display the preprocessed cropped image
            cv2.imshow('Preprocessed Cropped Image', preprocessed_image)
            cv2.waitKey(0)

        # Save the image with detected bounding boxes
        detected_image_path = os.path.join(save_dir, "detected_" + image_filename)
        cv2.imwrite(detected_image_path, frame)
        print(f"Image with detected objects saved as '{detected_image_path}'")

        # Display the image with detected bounding boxes
        cv2.imshow('Detected Image', frame)
        cv2.waitKey(0)
    else:
        print("Error: Could not read frame.")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

