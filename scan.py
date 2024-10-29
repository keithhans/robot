import time
import numpy as np
import matplotlib.pyplot as plt
from pymycobot import MyCobot
import datetime
import traceback

# 机械臂的工作范围
ARM_X_MIN = 160
ARM_X_MAX = 230
ARM_Y_MIN = -100
ARM_Y_MAX = 100
ARM_Z_DOWN = 80  # 假设Z轴高度固定
STEP = 40  # 扫描间隔

def generate_scan_points():
    """生成扫描点的坐标"""
    points = []
    for x in range(ARM_X_MIN, ARM_X_MAX + STEP, STEP):
        for y in range(ARM_Y_MIN, ARM_Y_MAX + STEP, STEP):
            points.append([x, y, ARM_Z_DOWN, -175, 0, -90])  # 固定姿态
    return np.array(points)

def plot_position_errors(target_positions, actual_positions):
    """绘制位置误差图"""
    # 计算误差
    errors = actual_positions - target_positions
    
    # 创建误差向量场图 (x-y平面)
    plt.figure(figsize=(12, 8))
    plt.quiver(target_positions[:, 0], target_positions[:, 1], 
              errors[:, 0], errors[:, 1], 
              angles='xy', scale_units='xy', scale=0.1)
    plt.plot(target_positions[:, 0], target_positions[:, 1], 'ro', label='Target Positions')
    plt.plot(actual_positions[:, 0], actual_positions[:, 1], 'bo', label='Actual Positions')
    
    plt.title('Position Errors in X-Y Plane (Arrows show direction and magnitude of error)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('position_errors_xy.png')
    plt.close()
    
    # 绘制误差直方图
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.hist(errors[:, 0], bins=20)
    plt.title('X Error Distribution')
    plt.xlabel('Error (mm)')
    plt.ylabel('Count')
    
    plt.subplot(132)
    plt.hist(errors[:, 1], bins=20)
    plt.title('Y Error Distribution')
    plt.xlabel('Error (mm)')
    plt.ylabel('Count')
    
    plt.subplot(133)
    plt.hist(errors[:, 2], bins=20)
    plt.title('Z Error Distribution')
    plt.xlabel('Error (mm)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()
    
    # 绘制3D误差散点图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
              c='r', marker='o', label='Target')
    ax.scatter(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 
              c='b', marker='^', label='Actual')
    
    # 添加误差线
    for i in range(len(target_positions)):
        ax.plot([target_positions[i, 0], actual_positions[i, 0]],
                [target_positions[i, 1], actual_positions[i, 1]],
                [target_positions[i, 2], actual_positions[i, 2]], 'k-', alpha=0.3)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Position Errors')
    ax.legend()
    plt.savefig('position_errors_3d.png')
    plt.close()
    
    # 计算并打印统计信息
    # 计算3D欧氏距离误差
    distance_errors = np.linalg.norm(errors, axis=1)
    mean_error = np.mean(distance_errors)
    max_error = np.max(distance_errors)
    std_error = np.std(distance_errors)
    
    print(f"\nError Statistics (3D):")
    print(f"Mean Error: {mean_error:.2f} mm")
    print(f"Max Error: {max_error:.2f} mm")
    print(f"Standard Deviation: {std_error:.2f} mm")
    
    # 分轴统计
    print("\nPer-Axis Statistics:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"\n{axis} Axis:")
        print(f"Mean Error: {np.mean(np.abs(errors[:, i])):.2f} mm")
        print(f"Max Error: {np.max(np.abs(errors[:, i])):.2f} mm")
        print(f"Std Error: {np.std(errors[:, i]):.2f} mm")
    
    return mean_error, max_error, std_error

def main():
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)
    
    # 生成扫描点
    target_points = generate_scan_points()
    actual_points = []
    
    try:
        print(f"Starting scan with {len(target_points)} points...")
        
        for i, point in enumerate(target_points):
            print(f"\nMoving to point {i+1}/{len(target_points)}")
            print(f"Target position: {point[:3]}")
            
            # 移动到目标位置
            mc.send_coords(point.tolist(), 50, 0)
            time.sleep(3)  # 等待机械臂稳定
            
            # 获取实际位置
            actual_coords = None
            retry_count = 0
            while actual_coords is None and retry_count < 3:
                actual_coords = mc.get_coords()
                if actual_coords is None:
                    retry_count += 1
                    time.sleep(0.1)
            
            if actual_coords is None:
                print(f"Warning: Could not get coordinates for point {i+1}")
                actual_coords = [0, 0, 0, 0, 0, 0]  # 使用默认值
            
            print(f"Actual position: {actual_coords[:3]}")
            actual_points.append(actual_coords)
        
        # 转换为numpy数组
        actual_points = np.array(actual_points)
        
        # 保存数据
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f'scan_results_{timestamp}.npz'
        np.savez(filename, 
                target_points=target_points,
                actual_points=actual_points)
        print(f"\nSaved scan results to {filename}")
        
        # 分析和绘制结果
        mean_error, max_error, std_error = plot_position_errors(
            target_points[:, :3], actual_points[:, :3])
        
        # 保存统计结果到文本文件
        with open(f'scan_statistics_{timestamp}.txt', 'w') as f:
            f.write(f"Scan Statistics:\n")
            f.write(f"Mean Error: {mean_error:.2f} mm\n")
            f.write(f"Max Error: {max_error:.2f} mm\n")
            f.write(f"Standard Deviation: {std_error:.2f} mm\n")
            f.write(f"\nScan Parameters:\n")
            f.write(f"X Range: {ARM_X_MIN} to {ARM_X_MAX} mm\n")
            f.write(f"Y Range: {ARM_Y_MIN} to {ARM_Y_MAX} mm\n")
            f.write(f"Z Height: {ARM_Z_DOWN} mm\n")
            f.write(f"Step Size: {STEP} mm\n")
        
    except KeyboardInterrupt:
        print("\nScan interrupted by user")
        mc.stop()
    except Exception as e:
        print(f"Error during scan: {e}")
        traceback.print_exc()
    finally:
        print("\nScan completed")
        mc.stop()

if __name__ == "__main__":
    main()
