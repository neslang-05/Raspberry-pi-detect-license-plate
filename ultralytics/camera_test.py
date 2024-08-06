import cv2

cap = cv2.VideoCapture(0)  # 0 represents the default camera
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Cannot read frame from camera")
else:
    print("Cannot open camera")