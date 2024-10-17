import cv2
import numpy as np
import time
import datetime
from pymycobot import MyCobot

def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    # 绕 Z 轴的旋转矩阵
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,            0,           1]])

    # 绕 Y 轴的旋转矩阵
    R_y = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                    [0,            1, 0           ],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    # 绕 X 轴的旋转矩阵
    R_x = np.array([[1, 0,            0           ],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),  np.cos(roll)]])

    # 计算最终的旋转矩阵
    R = R_z @ R_y @ R_x
    return R

def rotation_matrix_to_euler_angles(R):
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

# camera to world
roll_c2w = np.radians(-90)   # 以弧度表示的滚转 -90
pitch_c2w = np.radians(0)  # 以弧度表示的俯仰
yaw_c2w = np.radians(180)   # 以弧度表示的偏航 180

# 得到相机在世界坐标系中的旋转矩阵
R_cam2world = euler_angles_to_rotation_matrix(roll_c2w, pitch_c2w, yaw_c2w)

print(R_cam2world)

# camera to world
x = 0.5
y = 0.3
z = 0.3

t_cam2world = np.array([x, y, z])

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

            R_target2cam, _ = cv2.Rodrigues(rvec)
            t_target2cam = tvec.squeeze()

            # Calculate roll pitch yaw of R_target2cam
            rr, pp, yy = rotation_matrix_to_euler_angles(R_target2cam)

            # 计算目标在世界坐标系中的旋转矩阵
            R_target2world = R_cam2world @ R_target2cam

            # 计算目标在世界坐标系中的平移向量
            t_target2world = t_cam2world + R_cam2world @ t_target2cam

            # 计算目标在世界坐标系中的欧拉角
            roll, pitch, yaw = rotation_matrix_to_euler_angles(R_target2world)

            # Draw the axis for the marker
            cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, 0.02)

            # Display t values on the frame
            formatted_tvec = [f"{val:.3f}" for val in tvec[0][0]]

            formatted_tvec_t2w = [f"{val:.3f}" for val in t_target2world]

            # Display the marker ID and pose information on the frame
            text = f"id: {marker_ids[i]}, t2c tvec: {formatted_tvec}, r: {rr:.0f}, p: {pp:.0f}, y: {yy:.0f}. t2w t:{formatted_tvec_t2w} r: {roll:.0f}, p: {pitch:.0f}, y: {yaw:.0f}"
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
            start_tvec = t_target2world
            start_coods = mc.get_coords()
            tracking = True

    if tracking:
        # go to the target position
        print("go to target position")
        new_tvec = t_target2world
        move_tvec = new_tvec - start_tvec
        new_coods = [start_coods[0] + move_tvec[0] * 1000,
                     start_coods[1] + move_tvec[1] * 1000,
                     start_coods[2] + move_tvec[2] * 1000,
                     start_coods[3],
                     start_coods[4],
                     start_coods[5]]
        print("start_coods", start_coods)
        print("new coods", new_coods)
        #mc.send_coords(new_coods, 10, 1)    

cap.release()
cv2.destroyAllWindows()
