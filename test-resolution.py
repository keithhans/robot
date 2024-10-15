import cv2

cap = cv2.VideoCapture(0)

for res in [(320, 240), (640, 480), (1280, 720), (1920, 1080)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if width == res[0] and height == res[1]:
        print(f"supported resolution: {width}x{height}")

cap.release()