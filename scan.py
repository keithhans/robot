import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymycobot import MyCobot
import datetime
import traceback
from scipy.interpolate import griddata

# 机械臂的工作范围
ARM_X_MIN = 150
ARM_X_MAX = 250
ARM_Y_MIN = -100
ARM_Y_MAX = 100
ARM_Z_DOWN = 80  # 假设Z轴高度固定
STEP = 20  # 扫描间隔

def generate_scan_points():
    """生成扫描点的坐标"""
    points = []
    for x in range(ARM_X_MIN, ARM_X_MAX + 1, STEP):
        for y in range(ARM_Y_MIN, ARM_Y_MAX + 1, STEP):
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
    
    # 绘制3D误差散点图 - 交互式显示
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
    ax.set_title('3D Position Errors\n(Use mouse to rotate)')
    ax.legend()
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 添加网格
    ax.grid(True)
    
    # 设置坐标轴比例相同
    max_range = np.array([
        target_positions[:, 0].max() - target_positions[:, 0].min(),
        target_positions[:, 1].max() - target_positions[:, 1].min(),
        target_positions[:, 2].max() - target_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (target_positions[:, 0].max() + target_positions[:, 0].min()) * 0.5
    mid_y = (target_positions[:, 1].max() + target_positions[:, 1].min()) * 0.5
    mid_z = (target_positions[:, 2].max() + target_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存静态图
    plt.savefig('position_errors_3d.png')
    
    # 显示交互式图形
    print("\n正在显示3D交互图形...")
    print("使用鼠标拖动来旋转视角")
    print("使用鼠标滚轮来缩放")
    print("关闭图形窗口以继续程序")
    plt.show()
    
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

def predict_position_errors(target_positions, actual_positions, grid_size=20):
    """预测任意位置的定位误差"""
    # 计算实际误差
    errors = actual_positions - target_positions
    
    # 创建预测用的网格点
    x_min, x_max = ARM_X_MIN, ARM_X_MAX
    y_min, y_max = ARM_Y_MIN, ARM_Y_MAX
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    
    # 对每个坐标分量进行插值
    error_predictions = []
    for i in range(3):  # x, y, z 三个方向的误差
        zi = griddata((target_positions[:, 0], target_positions[:, 1]), 
                     errors[:, i], 
                     (xi, yi), 
                     method='cubic',
                     fill_value=0)
        error_predictions.append(zi)
    
    # 绘制预测误差图
    fig = plt.figure(figsize=(15, 5))
    titles = ['X Error', 'Y Error', 'Z Error']
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        im = ax.imshow(error_predictions[i], 
                      extent=[x_min, x_max, y_min, y_max],
                      origin='lower',
                      aspect='auto',
                      cmap='RdYlBu')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Predicted {titles[i]} (mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        # 添加实际测量点
        ax.scatter(target_positions[:, 0], target_positions[:, 1], 
                  c='black', s=10, alpha=0.5, label='Measured Points')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('error_prediction_map.png')
    plt.close()
    
    # 绘制总误差预测图
    total_error = np.sqrt(np.sum(np.array(error_predictions)**2, axis=0))
    plt.figure(figsize=(10, 8))
    im = plt.imshow(total_error, 
                    extent=[x_min, x_max, y_min, y_max],
                    origin='lower',
                    aspect='auto',
                    cmap='viridis')
    plt.colorbar(im, label='Predicted Total Error (mm)')
    plt.title('Predicted Total Position Error')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.scatter(target_positions[:, 0], target_positions[:, 1], 
                c='red', s=10, alpha=0.5, label='Measured Points')
    plt.legend()
    plt.savefig('total_error_prediction_map.png')
    plt.close()
    
    return error_predictions, (xi, yi)

def predict_error_at_point(point, target_positions, errors, method='cubic'):
    """预测特定点的误差"""
    predicted_error = []
    for i in range(3):  # x, y, z 三个方向
        error_i = griddata((target_positions[:, 0], target_positions[:, 1]),
                          errors[:, i],
                          (point[0], point[1]),
                          method=method)
        predicted_error.append(float(error_i))
    return np.array(predicted_error)

def validate_predictions(target_positions, actual_positions, test_ratio=0.2):
    """验证预测误差的准确性"""
    # 随机选择测试点
    n_points = len(target_positions)
    n_test = max(1, int(n_points * test_ratio))  # 确保至少有一个测试点
    
    if n_points < 5:  # 如果数据点太少，无法进行有效验证
        print("Warning: Too few points for meaningful validation")
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    test_indices = np.random.choice(n_points, n_test, replace=False)
    train_indices = np.array([i for i in range(n_points) if i not in test_indices])
    
    # 分割训练集和测试集
    train_targets = target_positions[train_indices]
    train_actuals = actual_positions[train_indices]
    test_targets = target_positions[test_indices]
    test_actuals = actual_positions[test_indices]
    
    # 计算实际误差
    actual_errors = test_actuals - test_targets
    
    # 使用训练集预测测试集的误差
    predicted_errors = []
    errors = train_actuals - train_targets
    
    for point in test_targets:
        try:
            predicted_error = predict_error_at_point(point, train_targets, errors, method='linear')  # 改用线性插值
            if np.any(np.isnan(predicted_error)):
                # 如果预测值是 NaN，使用最近点的误差
                distances = np.linalg.norm(train_targets - point, axis=1)
                nearest_idx = np.argmin(distances)
                predicted_error = errors[nearest_idx]
            predicted_errors.append(predicted_error)
        except Exception as e:
            print(f"Warning: Prediction failed for point {point}: {e}")
            # 使用平均误差作为后备方案
            predicted_error = np.mean(errors, axis=0)
            predicted_errors.append(predicted_error)
    
    predicted_errors = np.array(predicted_errors)
    
    # 检查是否有有效的预测结果
    if len(predicted_errors) == 0:
        print("Warning: No valid predictions generated")
        return np.zeros(3), np.zeros(3), np.zeros(3)
    
    # 计算预测误差与实际误差的差异
    prediction_accuracy = np.abs(predicted_errors - actual_errors)
    
    # 绘制对比图
    plt.figure(figsize=(15, 5))
    titles = ['X Error', 'Y Error', 'Z Error']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.scatter(actual_errors[:, i], predicted_errors[:, i], alpha=0.5)
        plt.plot([-10, 10], [-10, 10], 'r--')  # 理想预测线
        plt.xlabel('Actual Error (mm)')
        plt.ylabel('Predicted Error (mm)')
        plt.title(f'{titles[i]} Prediction')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('error_prediction_validation.png')
    plt.close()
    
    # 计算统计信息
    mean_accuracy = np.mean(prediction_accuracy, axis=0)
    max_accuracy = np.max(prediction_accuracy, axis=0)
    rmse = np.sqrt(np.mean((predicted_errors - actual_errors)**2, axis=0))
    
    print("\nPrediction Validation Results:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"\n{axis} Axis:")
        print(f"Mean prediction error: {mean_accuracy[i]:.2f} mm")
        print(f"Max prediction error: {max_accuracy[i]:.2f} mm")
        print(f"RMSE: {rmse[i]:.2f} mm")
        print(f"Number of test points: {len(predicted_errors)}")
    
    return mean_accuracy, max_accuracy, rmse

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
        
        # 预测误差
        print("\nGenerating error prediction maps...")
        error_predictions, (xi, yi) = predict_position_errors(
            target_points[:, :3], actual_points[:, :3])
        
        # 测试预测功能
        test_points = [
            [200, 0, ARM_Z_DOWN],  # 中心点
            [ARM_X_MIN, ARM_Y_MIN, ARM_Z_DOWN],  # 左下角
            [ARM_X_MAX, ARM_Y_MAX, ARM_Z_DOWN],  # 右上角
        ]
        
        print("\nPredicted errors at test points:")
        errors = actual_points[:, :3] - target_points[:, :3]
        for point in test_points:
            predicted_error = predict_error_at_point(point, target_points[:, :3], errors)
            print(f"\nPoint {point[:2]} mm:")
            print(f"Predicted error: X={predicted_error[0]:.2f}, Y={predicted_error[1]:.2f}, Z={predicted_error[2]:.2f} mm")
            print(f"Total error magnitude: {np.linalg.norm(predicted_error):.2f} mm")
        
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
            f.write("\nPredicted Errors at Test Points:\n")
            for i, point in enumerate(test_points):
                predicted_error = predict_error_at_point(point, target_points[:, :3], errors)
                f.write(f"\nTest Point {i+1} ({point[0]}, {point[1]}):\n")
                f.write(f"Predicted error: X={predicted_error[0]:.2f}, Y={predicted_error[1]:.2f}, Z={predicted_error[2]:.2f} mm\n")
                f.write(f"Total error magnitude: {np.linalg.norm(predicted_error):.2f} mm\n")
        
        # 验证预测准确性
        print("\nValidating prediction accuracy...")
        mean_accuracy, max_accuracy, rmse = validate_predictions(
            target_points[:, :3], actual_points[:, :3])
        
        # 将验证结果也保存到统计文件中
        with open(f'scan_statistics_{timestamp}.txt', 'a') as f:
            f.write("\nPrediction Validation Results:\n")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                f.write(f"\n{axis} Axis:\n")
                f.write(f"Mean prediction error: {mean_accuracy[i]:.2f} mm\n")
                f.write(f"Max prediction error: {max_accuracy[i]:.2f} mm\n")
                f.write(f"RMSE: {rmse[i]:.2f} mm\n")
    
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
