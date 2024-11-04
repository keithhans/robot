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

    print("Starting trajectory execution...")
    sample_time = t[1] - t[0]  # 采样时间间隔
    
    pen_down = 5 #35

    try:
        if args.pos:
            print("pos mode")
            for i in range(len(x)):
                start_time = time.time()
                coords[0] = x[i] * 1000 - pen_down
                coords[1] = y[i] * 1000 - pen_down
                coords[2] = z[i] * 1000 - pen_down
                mc.send_coords(coords, 50, 1)
                # 等待到下一个采样时刻
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.3f} {coords}")
                if elapsed_time < sample_time:
                    time.sleep(sample_time - elapsed_time)
        else:
            print("angle mode")
            for angle in joint_angles:
                start_time = time.time()
                mc.send_radians(angle, 50)            
                # 等待到下一个采样时刻
                elapsed_time = time.time() - start_time
                print(f"{elapsed_time:.3f} {angle}")
                if elapsed_time < sample_time:
                    time.sleep(sample_time - elapsed_time)
            
    except KeyboardInterrupt:
        print("\nTrajectory execution interrupted by user")

    print("Trajectory execution completed")

if __name__ == "__main__":
    main()
