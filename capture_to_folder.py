import cv2
import os

# URL of the IP camera stream
# Replace 'your_ip' and 'port' with the IP and port provided by DroidCam or your IP camera.
stream_url = "http://192.168.29.207:4747/video"

# Specify the directory where the image will be saved
save_dir = "captured_images"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Filename for the captured image
image_filename = "captured_image.png"

# Full path for the saved image
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
        # Display the captured frame
        cv2.imshow('Captured Image', frame)

        # Save the captured frame as a PNG image file in the specified directory
        cv2.imwrite(image_path, frame)

        print(f"Image captured and saved as '{image_path}'")

        # Wait for a key press and close the display window
        cv2.waitKey(0)
    else:
        print("Error: Could not read frame.")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
