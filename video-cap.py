import cv2

file_counter = 0

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
        filename = f"obs_{file_counter}.jpg"
        cv2.imwrite(filename, frame)
        file_counter += 1

cap.release()
cv2.destroyAllWindows()