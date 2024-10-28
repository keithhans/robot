import time
from pymycobot import MyCobot
import numpy as np
import matplotlib.pyplot as plt

def test_joint_velocity(mc, joint_id, target_angle=90):
    """
    测试单个关节的速度
    joint_id: 1-6
    target_angle: 目标角度（度）
    """
    try:
        # 记录起始角度
        start_angles = mc.get_angles()
        start_angle = start_angles[joint_id-1]
        print(f"Joint {joint_id} starting angle: {start_angle}")
        
        # 记录角度和时间
        angles = [start_angle]
        times = [0]
        start_time = time.time()
        
        # 开始运动
        direction = 1 if target_angle > start_angle else 0
        mc.jog_angle(joint_id, direction, 10)  # 使用最大速度100
        
        # 持续记录直到达到目标角度或超时
        timeout = 10  # 10秒超时
        while time.time() - start_time < timeout:
            current_angles = mc.get_angles()
            while current_angles == None:
                current_angles = mc.get_angles()
            print(current_angles)
            current_angle = current_angles[joint_id-1]
            current_time = time.time() - start_time
            
            angles.append(current_angle)
            times.append(current_time)
            
            # 检查是否达到目标角度
            if abs(current_angle) >= abs(target_angle):
                break
            
            time.sleep(0.1)  # 采样间隔100ms
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        mc.stop()  # 停止机械臂
        raise  # 重新抛出异常，让主程序处理
    finally:
        # 无论如何都确保机械臂停止
        mc.stop()
    
    # 计算平均角速度
    total_angle_change = abs(angles[-1] - angles[0])
    total_time = times[-1]
    average_velocity = total_angle_change / total_time if total_time > 0 else 0
    
    print(f"\nJoint {joint_id} Results:")
    print(f"Total angle change: {total_angle_change:.2f} degrees")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average velocity: {average_velocity:.2f} degrees/s")
    print(f"                 {np.radians(average_velocity):.2f} rad/s")
    
    # 绘制角度-时间图
    plt.figure(figsize=(10, 6))
    plt.plot(times, angles, 'b-', label=f'Joint {joint_id}')
    plt.title(f'Joint {joint_id} Angle vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'joint_{joint_id}_velocity_test.png')
    plt.close()
    
    return average_velocity

def main():
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.set_fresh_mode(0)
    
    try:
        
        results = []
        # 测试每个关节
        for joint_id in range(1, 7):
            print(f"\nTesting Joint {joint_id}...")
            input(f"Press Enter to start testing joint {joint_id}...")
            
            # 确保从零位开始
            print("Moving to home position...")
            mc.send_angles([0, 0, 0, 0, 0, 0], 50)
            time.sleep(3)
            
            # 测试关节速度
            velocity = test_joint_velocity(mc, joint_id)
            results.append(velocity)
            
            # 等待用户确认继续
            input("Press Enter to continue to next joint...")
        
        # 打印总结
        print("\nVelocity Test Results Summary:")
        print("Joint\tDeg/s\tRad/s")
        print("-" * 30)
        for i, velocity in enumerate(results, 1):
            print(f"{i}\t{velocity:.2f}\t{np.radians(velocity):.2f}")
        
        # 保存结果到文件
        np.savez('joint_velocity_calibration.npz', 
                 velocities=np.array(results),
                 velocities_rad=np.radians(results))
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        mc.stop()  # 确保机械臂停止
    finally:
        # 无论如何都确保机械臂停止
        mc.stop()

if __name__ == "__main__":
    main()
