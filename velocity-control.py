import numpy as np
import argparse
import time
from pymycobot import MyCobot

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

def main():
    parser = argparse.ArgumentParser(description='Load and process circle trajectory data.')
    parser.add_argument('npz_file', help='Path to the NPZ file containing circle trajectory data')
    args = parser.parse_args()

    try:
        data = np.load(args.npz_file)
        start_position = data['start_position'] * 1000
        start_rpy = data['start_rpy'] / np.pi * 180
        joint_velocities = data['joint_velocities']
        t = data['t']
        
        print("Loaded data from:", args.npz_file)
        print("Start position:", start_position)
        print("Start RPY:", start_rpy)
        print("Joint velocities shape:", joint_velocities.shape)
        print("Time points:", len(t))
        
    except Exception as e:
        print(f"Error loading the NPZ file: {e}")
        return
   
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)

    # move robot to initial position
    initial_coords = np.concatenate([start_position, start_rpy])
    mc.send_coords(initial_coords.tolist(), 50, 1)
    time.sleep(10)

    print("Starting trajectory execution...")
    sample_time = t[1] - t[0]  # 采样时间间隔

    try:
        for velocities in joint_velocities:
            start_time = time.time()
            
            # 对每个关节执行速度命令
            for joint_id in range(6):
                direction, speed = velocity_to_jog_command(velocities[joint_id], joint_id)
                print(f"joint {joint_id} speed {speed} direction {direction}")
                mc.jog_angle(joint_id + 1, direction, speed)  # joint_id从1开始
            
            # 等待到下一个采样时刻
            elapsed_time = time.time() - start_time
            if elapsed_time < sample_time:
                time.sleep(sample_time - elapsed_time)
            
    except KeyboardInterrupt:
        print("\nTrajectory execution interrupted by user")
        # 确保所有关节都停止
        mc.stop()

    mc.stop()
    print("Trajectory execution completed")

if __name__ == "__main__":
    main()
