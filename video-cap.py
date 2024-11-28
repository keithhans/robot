import cv2
import time
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Capture video from camera')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    args = parser.parse_args()

    # Open video capture
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(f"Started capturing from camera {args.camera}")
    print("Press 's' to save image")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame for capturing
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.time()
            dt_object = datetime.datetime.fromtimestamp(timestamp)
            formatted_time = dt_object.strftime('%Y-%m-%d-%H-%M-%S')
            filename = f"obs_{formatted_time}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved image to {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
