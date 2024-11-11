import numpy as np
import argparse
import time
from pymycobot import MyCobot
import matplotlib.pyplot as plt

def velocity_to_jog_command(velocity, joint_id):
    """Convert joint velocity to direction and speed for jog_angle command"""
    # handle weird behavior of jog_angle
    direction = 0
    if joint_id == 3:
        direction = 1 if velocity >= 0 else 0
    else:
        direction = 1 if velocity <= 0 else 0
    # 将速度映射到1-100的范围
    # 假设最大速度为180 degree/s  
    # https://docs.elephantrobotics.com/docs/mycobot_280_pi_cn/4-SupportAndService/9.Troubleshooting/9.3-hardware.html
    MAX_VELOCITY = np.pi
    speed = int(min(100, max(1, abs(velocity) / MAX_VELOCITY * 100)))
    return direction, speed

def calculate_tracking_metrics(target_angles, actual_angles):
    """计算跟踪效果的统计指标"""
    # 过滤掉 None 值和大于 3.14 的值
    valid_pairs = [(target, actual) for target, actual in zip(target_angles, actual_angles) 
                  if actual is not None and all(abs(a) <= 3.14 for a in actual)]
    if not valid_pairs:
        print("No valid data for metrics calculation")
        return
    
    valid_targets, valid_actuals = zip(*valid_pairs)
    valid_targets = np.array(valid_targets)
    valid_actuals = np.array(valid_actuals)
    
    # 计算每个关节的指标
    metrics = {}
    for joint in range(6):
        target = valid_targets[:, joint]
        actual = valid_actuals[:, joint]
        error = actual - target
        
        metrics[f'joint_{joint+1}'] = {
            'mean_error': np.mean(np.abs(error)),  # 平均绝对误差
            'max_error': np.max(np.abs(error)),    # 最大绝对误差
            'rmse': np.sqrt(np.mean(error**2)),    # 均方根误差
            'std_error': np.std(error),            # 误差标准差
            'tracking_ratio': len(valid_pairs) / len(target_angles) * 100  # 有效数据比例
        }
    
    # 保存指标到文件
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    with open(f'tracking_metrics_{timestamp}.txt', 'w') as f:
        f.write("Tracking Performance Metrics:\n")
        f.write("===========================\n\n")
        
        for joint, joint_metrics in metrics.items():
            f.write(f"{joint}:\n")
            f.write(f"  Mean Absolute Error: {joint_metrics['mean_error']:.4f} rad\n")
            f.write(f"  Max Absolute Error: {joint_metrics['max_error']:.4f} rad\n")
            f.write(f"  RMSE: {joint_metrics['rmse']:.4f} rad\n")
            f.write(f"  Standard Deviation: {joint_metrics['std_error']:.4f} rad\n")
            f.write(f"  Valid Data Ratio: {joint_metrics['tracking_ratio']:.1f}%\n\n")
        
        # 计算总体指标
        total_mean_error = np.mean([m['mean_error'] for m in metrics.values()])
        total_max_error = np.max([m['max_error'] for m in metrics.values()])
        total_rmse = np.sqrt(np.mean([m['rmse']**2 for m in metrics.values()]))
        
        f.write("Overall Performance:\n")
        f.write(f"  Average Mean Error: {total_mean_error:.4f} rad\n")
        f.write(f"  Maximum Error: {total_max_error:.4f} rad\n")
        f.write(f"  Overall RMSE: {total_rmse:.4f} rad\n")
    
    return metrics

def plot_angle_comparison(target_angles, actual_angles):
    """绘制目标角度和实际角度的对比图，忽略 None 值和大于 3.14 的值"""
    # 过滤掉 None 值和大于 3.14 的值
    valid_pairs = [(target, actual) for target, actual in zip(target_angles, actual_angles) 
                  if actual is not None and all(abs(a) <= 3.14 for a in actual)]
    if not valid_pairs:  # 如果没有有效数据
        print("No valid angle data for plotting")
        return
        
    # 分离有效的目标角度和实际角度
    valid_targets, valid_actuals = zip(*valid_pairs)
    valid_targets = np.array(valid_targets)
    valid_actuals = np.array(valid_actuals)
    
    # 创建6个子图，每个关节一个
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Joint Angles: Target vs Actual')
    
    for joint in range(6):
        row = joint // 2
        col = joint % 2
        
        axs[row, col].plot(valid_targets[:, joint], 'b-', label='Target', alpha=0.7)
        axs[row, col].plot(valid_actuals[:, joint], 'r--', label='Actual', alpha=0.7)
        axs[row, col].set_title(f'Joint {joint+1}')
        axs[row, col].set_xlabel('Sample')
        axs[row, col].set_ylabel('Angle (rad)')
        axs[row, col].legend()
        axs[row, col].grid(True)
    
    # 在图上添加统计指标
    metrics = calculate_tracking_metrics(target_angles, actual_angles)
    for joint in range(6):
        row = joint // 2
        col = joint % 2
        joint_metrics = metrics[f'joint_{joint+1}']
        stats_text = f"RMSE: {joint_metrics['rmse']:.4f}\nMean Error: {joint_metrics['mean_error']:.4f}"
        axs[row, col].text(0.02, 0.98, stats_text,
                          transform=axs[row, col].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    # 保存图像到文件
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f'angle_comparison_{timestamp}.png')
    plt.close()  # 关闭图形，不显示

def main():
    parser = argparse.ArgumentParser(description='Load and process trajectory data.')
    parser.add_argument('npz_file', help='Path to the NPZ file containing trajectory data')
    parser.add_argument('-p', '--pos', action='store_true', help='Use ee position')

    args = parser.parse_args()

    try:
        data = np.load(args.npz_file)
        start_position = data['start_position'] * 1000
        start_rpy = data['start_rpy'] / np.pi * 180
        joint_velocities = data['joint_velocities']
        joint_angles = data['joint_angles']
        x = data['x']
        y = data['y']
        z = data['z']
        t = data['t']
        
        print("Loaded data from:", args.npz_file)
        print("Start position:", start_position)
        print("Start RPY:", start_rpy)
        print("Joint velocities shape:", joint_velocities.shape)
        print("Joint angles shape:", joint_angles.shape)
        print("Time points:", len(t))
        
    except Exception as e:
        print(f"Error loading the NPZ file: {e}")
        return
   
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)

    # move robot to initial position
    initial_coords = np.concatenate([start_position, start_rpy])
    coords = initial_coords.tolist()
    mc.send_coords(coords, 50, 1)
    time.sleep(5)

    sample_time = t[1] - t[0]  # 采样时间间隔
    sample_time = 0.20
    print(f"Starting trajectory execution... sample interval:{sample_time}")

    pen_down = 0 #38
    
    if args.pos:
        # pen down
        coords[2] -= pen_down
        mc.send_coords(coords, 50, 1)
        time.sleep(2)
    

    try:
        if args.pos:
            print("pos mode")
            for i in range(len(x)):
                start_time = time.time()
                coords[0] = x[i] * 1000 
                coords[1] = y[i] * 1000
                coords[2] = z[i] * 1000 - pen_down
                mc.send_coords(coords, 50, 1)
                # 等待到下一个采样时刻
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.3f} {coords}")
                if elapsed_time < sample_time:
                    time.sleep(sample_time - elapsed_time)
            mc.send_coord(3, coords[2] + pen_down, 50)  # pen up
        else:
            print("angle mode")
            actual_angles = []  # 存储实际角度
            
            for angle in joint_angles:
                start_time = time.time()
                mc.send_radians(angle, 50)            
                time.sleep(0.18)
                
                try:
                    current = mc.get_radians()
                except TypeError as e:
                    print(f"{e}")
                    current = None
                
                actual_angles.append(current)
                
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.3f} target:{angle} current:{current}")
                
                if elapsed_time < sample_time:
                    time.sleep(sample_time - elapsed_time)
            
            # 计算和绘制跟踪效果
            metrics = calculate_tracking_metrics(joint_angles, actual_angles)
            plot_angle_comparison(joint_angles, actual_angles)
            
    except KeyboardInterrupt:
        print("\nTrajectory execution interrupted by user")
    finally:
        print("Trajectory execution completed")

if __name__ == "__main__":
    main()
