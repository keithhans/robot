import cv2
import numpy as np
import time
import datetime


def rotation_vector_to_euler_angles(r_vec):
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(r_vec)
    
    # 从旋转矩阵提取欧拉角（假设使用 ZYX 顺序）
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  # 计算 sy

    singular = sy < 1e-6  # 判断是否接近奇异点

    if not singular:
        x_rotation = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y_rotation = np.arctan2(-R[2, 0], sy)       # Pitch
        z_rotation = np.arctan2(R[1, 0], R[0, 0])   # Yaw
    else:
        x_rotation = np.arctan2(-R[1, 2], R[1, 1])
        y_rotation = np.arctan2(-R[2, 0], sy)
        z_rotation = 0

    return np.degrees([x_rotation, y_rotation, z_rotation])  # 返回 Roll, Pitch, Yaw


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
            formatted_rvec = [f"{val:.2f}" for val in rvec[0][0]]
            formatted_tvec = [f"{val:.3f}" for val in tvec[0][0]]
            roll, pitch, yaw = rotation_vector_to_euler_angles(rvec)
            text = f"id: {marker_ids[i]}, tvec: {formatted_tvec}, rvec: {formatted_rvec}, r: {roll:.0f}, p: {pitch:.0f}, y: {yaw:.0f}"
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
        dt_object = datetime.datetime.fromtimestamp(end_time)
        formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        filename = f"pose_{formatted_time}.jpg"
        cv2.imwrite(filename, frame)

cap.release()
cv2.destroyAllWindows()
