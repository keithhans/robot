import cv2
import numpy as np
import time
import datetime
from pymycobot import MyCobot

def euler_to_rotation_matrix(roll, pitch, yaw):
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

def coords2vector(coords):
    t_vec = np.array(coords[:3]) / 1000
    roll, pitch, yaw = np.radians(np.array(coords[3:]))
    R = euler_to_rotation_matrix()
    return R, t_vec


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

# Initialize the MyCobot
mc = MyCobot("/dev/ttyAMA0", 1000000)

# Create empty lists to store the data
p1 = []
p2 = []
p3 = []
p4 = []

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
    elif key == ord('c'):   # collect data
        print(rvec, tvec)
        print(rvec.shape, tvec.shape)
        R_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.squeeze()
        print(R_target2cam)
        print(t_target2cam)

        # collect bot data
        coords = mc.get_coords()
        print(coords)

        R_gripper2base, t_gripper2base = coords2vector(coords)
        print(R_gripper2base)
        print(t_gripper2base)

        p1.append(R_gripper2base)
        p2.append(t_gripper2base)
        p3.append(R_target2cam)
        p4.append(t_target2cam)
    elif key == ord('r'):   # reset data
        if len(p1) < 2:
            print("not enough data")
            continue
        R, t = cv2.calibrateHandEye(p1, p2, p3, p4)
        rvec, _ = cv2.Rodrigues(R)
        roll, pitch, yaw = rotation_vector_to_euler_angles(rvec)

        print("R:", R)
        print("t:",t)

        print("r p y: ",roll, pitch, yaw)

        p1 = []
        p2 = []
        p3 = []
        p4 = []


cap.release()
cv2.destroyAllWindows()
