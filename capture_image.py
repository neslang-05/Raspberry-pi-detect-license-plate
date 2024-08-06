import cv2

# URL of the IP camera stream
# Replace 'your_ip' and 'port' with the IP and port provided by DroidCam or your IP camera.
stream_url = "http://192.168.29.207:4747/video"

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

        # Save the captured frame as an image file
        cv2.imwrite('captured_image.jpg', frame)

        print("Image captured and saved as 'captured_image.jpg'")

        # Wait for a key press and close the display window
        cv2.waitKey(0)
    else:
        print("Error: Could not read frame.")

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

