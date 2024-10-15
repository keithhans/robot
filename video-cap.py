import cv2
import time
import datetime


# Open video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame for capturing
    cv2.imshow("Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = time.time()

        dt_object = datetime.datetime.fromtimestamp(timestamp)
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        filename = f"obs_{formatted_time}.jpg"
        cv2.imwrite(filename, frame)

cap.release()
cv2.destroyAllWindows()