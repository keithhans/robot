import cv2
import numpy as np
import time
import datetime
from pymycobot import MyCobot

# Load the predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Load the camera matrix and distortion coefficients (from calibration)
calibration_data = np.load('camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
distortion_coeffs = calibration_data['distortion_coeffs']

# Open video capture
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the MyCobot
mc = MyCobot("/dev/ttyAMA0", 1000000)

tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_time = time.time()

    # Detect markers in the frame
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

    # Draw the detected markers
    cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    # Check if any markers were detected
    if marker_ids is not None:
        for i in range(len(marker_ids)):
            # Estimate pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 0.02, camera_matrix, distortion_coeffs)

            # Draw the axis for the marker
            cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, 0.02)

            # Display rvec and tvec values on the frame
            formatted_rvec = [f"{val:.3f}" for val in rvec[0][0]]
            formatted_tvec = [f"{val:.3f}" for val in tvec[0][0]]
            text = f"id: {marker_ids[i]}, tvec: {formatted_tvec}, rvec: {formatted_rvec}"
            cv2.putText(frame, text, (10, 60+30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    end_time = time.time()
    elapsed_time = end_time - start_time
    text = f"time elapsed: {elapsed_time:.3f}s"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the frame with detected markers and pose axes
    cv2.imshow("Pose Estimation", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if tracking:
            print("stop tracking")
            tracking = False
        else:
            # start tracking
            print("start tracking")
            start_tvec = tvec.squeeze()
            start_coods = mc.get_coords()
            tracking = True
            #R_target2cam, _ = cv2.Rodrigues(rvec)
            #t_target2cam = tvec.squeeze()

    if tracking:
        # go to the target position
        print("go to target position")
        new_tvec = tvec.squeeze()
        move_tvec = new_tvec - start_tvec
        new_coods = [start_coods[0] + move_tvec[0] * 1000,
                     start_coods[1] + move_tvec[1] * 1000,
                     start_coods[2] + move_tvec[2] * 1000,
                     start_coods[3],
                     start_coods[4],
                     start_coods[5]]
        print("start_coods", start_coods)
        print("new coods", new_coods)
        mc.send_coords(new_coods, 10, 1)    

cap.release()
cv2.destroyAllWindows()
