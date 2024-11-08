import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pymycobot import MyCobot

def generate_sine_trajectory(duration, sample_time=0.03):
    """生成正弦轨迹"""
    t = np.arange(0, duration, sample_time)
    # 频率为 0.2Hz，幅度为 45 度（约 0.785 弧度）
    angles = np.pi/4 / 3 * np.sin(2 * np.pi * 0.2 * t)  
    return t, angles

def plot_tracking_results(joint_id, sample_time, times, target_angles, actual_angles):
    """绘制跟踪结果"""
    # 过滤掉 None 值和异常值
    valid_data = [(t, target, actual) for t, target, actual in zip(times, target_angles, actual_angles)
                 if actual is not None and abs(actual) <= np.pi]
    
    if not valid_data:
        print("No valid data for plotting")
        return
    
    valid_times, valid_targets, valid_actuals = zip(*valid_data)
    
    # 计算跟踪误差指标
    errors = np.array(valid_actuals) - np.array(valid_targets)
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))
    mean_error = np.mean(np.abs(errors))
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制角度跟踪曲线
    plt.subplot(211)
    plt.plot(valid_times, valid_targets, 'b-', label='Target', alpha=0.7)
    plt.plot(valid_times, valid_actuals, 'r--', label='Actual', alpha=0.7)
    plt.title('Joint Angle Tracking')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    
    # 添加误差指标文本
    stats_text = f'RMSE: {rmse:.4f} rad\nMax Error: {max_error:.4f} rad\nMean Error: {mean_error:.4f} rad'
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 绘制误差曲线
    plt.subplot(212)
    plt.plot(valid_times, errors, 'g-', label='Error')
    plt.title('Tracking Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
    plt.savefig(f'servo_test_joint_{joint_id}_{sample_time}_{timestamp}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test joint servo performance')
    parser.add_argument('joint', type=int, choices=range(1, 7), 
                       help='Joint number (1-6)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Test duration in seconds')
    args = parser.parse_args()
    
    # 初始化机械臂
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)
    
    # 生成轨迹
    sample_time = 0.02
    times, target_angles = generate_sine_trajectory(args.duration, sample_time)
    actual_angles = []
    recorded_times = []
    
    print(f"Testing joint {args.joint} for {args.duration} seconds...")
    print("Press Ctrl+C to stop")
    
    try:
        # 先回到零位
        mc.send_radians([0, 0, 0, 0, 0, 0], 50)
        time.sleep(3)
        
        start_time = time.time()
        for t, angle in zip(times, target_angles):
            loop_start = time.time()
            
            # 构造完整的关节角度列表（其他关节保持为0）
            angles = [0] * 6
            angles[args.joint - 1] = angle
            
            # 发送角度命令
            mc.send_radians(angles, 100)
            
            # 读取实际角度
            try:
                time.sleep(0.01)
                current_angles = mc.get_radians()
                if current_angles is not None:
                    actual_angle = current_angles[args.joint - 1]
                else:
                    actual_angle = None
            except Exception as e:
                print(f"Error reading angles: {e}")
                actual_angle = None
            
            actual_angles.append(actual_angle)
            recorded_times.append(time.time() - start_time)
            
            # 等待到下一个采样时刻
            elapsed = time.time() - loop_start
            if elapsed < sample_time:
                time.sleep(sample_time - elapsed)
            
            # 打印进度
            print(f"Time: {recorded_times[-1]:.3f}s, Target: {angle:.4f}, Actual: {actual_angle}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # 停止机械臂
        mc.stop()
        
        # 回到零位
        mc.send_radians([0, 0, 0, 0, 0, 0], 50)
        
        # 绘制结果
        plot_tracking_results(args.joint, sample_time, recorded_times, target_angles, actual_angles)
        
        print("Test completed")

if __name__ == "__main__":
    main()
