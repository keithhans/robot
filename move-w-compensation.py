import numpy as np
from scipy.interpolate import griddata
from pymycobot import MyCobot
import argparse
import time

def predict_error_at_point(point, target_positions, errors, method='linear'):
    """预测特定点的误差"""
    predicted_error = []
    for i in range(3):  # x, y, z 三个方向
        error_i = griddata((target_positions[:, 0], target_positions[:, 1]),
                          errors[:, i],
                          (point[0], point[1]),
                          method=method)
        if np.isnan(error_i):
            # 如果预测值是 NaN，使用最近点的误差
            distances = np.linalg.norm(target_positions[:, :2] - np.array([point[0], point[1]]), axis=1)
            nearest_idx = np.argmin(distances)
            error_i = errors[nearest_idx, i]
        predicted_error.append(float(error_i))
    return np.array(predicted_error)

def move_with_compensation(mc, x, y, z, scan_data_file):
    """移动到指定位置，并对误差进行补偿"""
    # 加载扫描数据
    data = np.load(scan_data_file)
    target_points = data['target_points']
    actual_points = data['actual_points']
    
    # 计算历史误差
    errors = actual_points[:, :3] - target_points[:, :3]
    
    # 预测当前位置的误差
    point = np.array([x, y, z])
    predicted_error = predict_error_at_point(point, target_points[:, :3], errors)
    
    print(f"Target position: [{x}, {y}, {z}]")
    print(f"Predicted error: {predicted_error}")
    
    # 计算补偿后的位置
    compensated_position = point - predicted_error
    print(f"Compensated position: {compensated_position}")
    
    # 移动到补偿后的位置
    coords = [
        compensated_position[0],  # x
        compensated_position[1],  # y
        compensated_position[2],  # z
        -175,  # rx (与scan.py保持一致)
        0,     # ry
        -90    # rz
    ]
    
    print("Moving to compensated position...")
    mc.send_coords(coords, 50, 0)
    time.sleep(3)  # 等待机械臂到达位置
    
    # 获取实际位置
    actual_coords = None
    retry_count = 0
    while actual_coords is None and retry_count < 3:
        actual_coords = mc.get_coords()
        if actual_coords is None:
            retry_count += 1
            time.sleep(0.1)
    
    if actual_coords is None:
        print("Warning: Could not get actual coordinates")
        return None
    
    print(f"Actual position: {actual_coords[:3]}")
    
    # 计算实际误差
    actual_error = np.array(actual_coords[:3]) - point
    print(f"Actual error: {actual_error}")
    print(f"Prediction accuracy: {np.abs(actual_error - predicted_error)}")
    
    return actual_coords

def main():
    parser = argparse.ArgumentParser(description='Move robot with error compensation')
    parser.add_argument('scan_file', help='Path to the scan results NPZ file')
    parser.add_argument('x', type=float, help='Target X coordinate')
    parser.add_argument('y', type=float, help='Target Y coordinate')
    parser.add_argument('z', type=float, help='Target Z coordinate')
    args = parser.parse_args()
    
    # 初始化机械臂
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)
    
    try:
        # 移动到目标位置并进行误差补偿
        actual_coords = move_with_compensation(mc, args.x, args.y, args.z, args.scan_file)
        
        if actual_coords is not None:
            # 可以在这里添加更多的分析或可视化代码
            pass
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        mc.stop()
    except Exception as e:
        print(f"Error occurred: {e}")
        mc.stop()
    finally:
        print("Program completed")

if __name__ == "__main__":
    main()
