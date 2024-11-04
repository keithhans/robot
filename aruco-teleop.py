import cv2
import numpy as np
import time
import datetime
import threading
from pymycobot import MyCobot
from scipy.signal import butter, lfilter, lfilter_zi
import argparse

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

def butter_lowpass(cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butter_lowpass_lfilter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a)
    zi = zi * data[0]  # 使用信号的第一个值初始化zi
    y, _ = lfilter(b, a, data, zi=zi)
    return y[-1]

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)

lock = threading.Lock()
quiting = False

# 在全局变量区域添加一个新的列表来存储过滤后的坐标
filtered_coords_array = []

# 在全局变量区域添加新的变量
coords_array = []
new_coords = None

def control_thread():
    global tracking, new_coords, mc, coord_buffers, quiting, filtered_coords_array, coords_array, cutoff_frequency
    
    start_time = time.time()
   
    if cutoff_frequency > 0:
        # 为每个坐标分量初始化zi
        zi = {key: None for key in ['x', 'y', 'z', 'rx', 'ry', 'rz']}
        b, a = butter_lowpass(cutoff_frequency, sampling_freq, filter_order)
    
    while not quiting:
        if tracking and new_coords is not None:
            with lock:
                # 保存原始坐标
                coords_array.append(new_coords)

                if cutoff_frequency > 0:
                    # 更新缓冲区
                    filtered_coords = []
                    for i, key in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
                        coord_buffers[key].append(new_coords[i])
                        if len(coord_buffers[key]) > buffer_size:
                            coord_buffers[key].pop(0)

                        if len(coord_buffers[key]) == buffer_size:
                            if zi[key] is None:
                                zi[key] = lfilter_zi(b, a)
                                zi[key] = zi[key] * coord_buffers[key][0]
                            
                            filtered_value, zi[key] = lfilter(b, a, coord_buffers[key], zi=zi[key])
                            filtered_value = filtered_value[-1]
                            
                            # 对 x, y, z 和 rx, ry, rz 分别进行限制
                            if i < 3:  # x, y, z
                                filtered_value = clamp(filtered_value, -280, 280)
                            else:  # rx, ry, rz
                                filtered_value = clamp(filtered_value, -179.9, 179.9)
                            filtered_coords.append(round(filtered_value, 1))
                        else:
                            filtered_value = new_coords[i]
                            if i < 3:  # x, y, z
                                filtered_value = clamp(filtered_value, -280, 280)
                            else:  # rx, ry, rz
                                filtered_value = clamp(filtered_value, -179.9, 179.9)
                            filtered_coords.append(filtered_value)
                else:
                    # 如果 cutoff_frequency 为 0，直接使用原始坐标
                    filtered_coords = [clamp(val, -280 if i < 3 else -179.9, 280 if i < 3 else 179.9) for i, val in enumerate(new_coords)]

                # 发送坐标
                mc.send_coords(filtered_coords, 50, 1)
                print("coords sent", filtered_coords, " @", start_time)
                
                # 保存过滤后的坐标
                filtered_coords_array.append(filtered_coords)
        
        # 计算下一次调用的延迟
        elapsed_time = time.time() - start_time
        delay = max(0, 0.333 - elapsed_time)  # 确保至少延迟333ms
        time.sleep(delay)
        start_time = time.time()

# camera to world
roll_c2w = np.radians(-90)   # 以弧度表示的滚转 -90
pitch_c2w = np.radians(0)  # 以弧度表示的俯仰
yaw_c2w = np.radians(180)   # 以弧度表示的偏航 180

R_cam2world = euler_angles_to_rotation_matrix(roll_c2w, pitch_c2w, yaw_c2w)

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
print("fresh mode", mc.get_fresh_mode())

tracking = False
marker_ids = []

coords_array = []


sampling_freq = 10.0  # 假设的采样频率，根据实际情况调整
cutoff_frequency = 2.0  # 截止频率 (Hz)
filter_order = 3

buffer_size = 20  # 根据需要调整缓冲区大小

coord_buffers = {key: [] for key in ['x', 'y', 'z', 'rx', 'ry', 'rz']}



def main():
    global cutoff_frequency, tracking, coords_array, filtered_coords_array, new_coords, quiting

    last_time = time.time()

    parser = argparse.ArgumentParser(description='ArUco marker based robot control')
    parser.add_argument('-f', '--filter', type=float, default=2.0, help='Cutoff frequency for the low-pass filter (Hz). Set to 0 to disable filtering.')
    args = parser.parse_args()

    cutoff_frequency = args.filter

    # Start the control thread
    threading.Thread(target=control_thread).start()

    # Start the main loop
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        elapsed_time = time.time() - start_time
        # print(f"image captured. time elapsed: {elapsed_time:.3f}s")

        # Detect markers in the frame
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)

        # Draw the detected markers
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        # Check if any markers were detected
        if marker_ids is not None:
            for i in range(len(marker_ids)):
                # Estimate pose of the marker
                marker_size = 0.04  # 0.02
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners[i], 0.04, camera_matrix, distortion_coeffs)

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
        total_time = end_time - last_time
        text = f"time elapsed: {elapsed_time:.3f}s  total time:{total_time:.3f}s"
        last_time = end_time


        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Display the frame with detected markers and pose axes
        cv2.imshow("Pose Estimation", frame)

        elapsed_time = time.time() - start_time
        #print(f"img shown. time elapsed: {elapsed_time:.3f}s")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            quiting = True
            break
        elif key == ord('t'):
            if tracking:
                print("stop tracking")
                dt_object = datetime.datetime.fromtimestamp(time.time())
                formatted_time = dt_object.strftime('%Y-%m-%d-%H-%M-%S')
                filename = f"records_{formatted_time}.npz"
                np.savez(filename, coords_array=coords_array, filtered_coords_array=filtered_coords_array)
                coords_array = []
                filtered_coords_array = []  # 清空过滤后的坐标数组
                print("wrote coords to file")
                tracking = False
            else:
                # start tracking
                if marker_ids is None or len(marker_ids) == 0:
                    print("tracking can't start because no mark in sight")
                else:     
                    print("start tracking")
                    start_tvec = t_target2world
                    start_rMat = R_target2world
                    start_coords = mc.get_coords()
                    print("start_coords", start_coords)
                    origin_angles = np.array(start_coords[3:6])
                    origin_rMat = euler_angles_to_rotation_matrix(np.radians(start_coords[3]), np.radians(start_coords[4]), np.radians(start_coords[5]))
                    tracking = True

        if tracking:
            # go to the target position
            new_tvec = t_target2world
            move_tvec = (new_tvec - start_tvec) * 0.75
            new_rMat = R_target2world
            delta_r_Mat = new_rMat @ np.linalg.inv(start_rMat)
            final_r_Mat = delta_r_Mat @ origin_rMat
            new_angles = rotation_matrix_to_euler_angles(final_r_Mat)
            x = start_coords[0] + round(move_tvec[0] * 1000, 1)
            y = start_coords[1] + round(move_tvec[1] * 1000, 1)
            z = start_coords[2] + round(move_tvec[2] * 1000, 1)
            rx = round(new_angles[0], 2)
            ry = round(new_angles[1], 2)
            rz = round(new_angles[2], 2)
            
            if x <= 280 and x >= -280 and y <= 280 and y >= -280 and z <= 280 and z >= -280:
                with lock:
                    new_coords = [x, y, z, rx, ry, rz]
            else:
                print("Warning: x or y or z out of range", x, y, z)     # todo: figure out the root cause
            print("new coords", new_coords)
        
        elapsed_time = time.time() - start_time
        # print(f"total time elapsed: {elapsed_time:.3f}s")



    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
