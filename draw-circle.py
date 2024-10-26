import numpy as np
import matplotlib.pyplot as plt
from RobotArm import RobotArm

# Define the circle parameters
center = np.array([0.1, 0, 0.1])
radius = 0.05
total_time = 10  # seconds
sample_rate = 0.1  # 100ms

# Create a RobotArm instance (using example DH parameters)
d = [0.13156, 0, 0, 0.06462, 0.07318, 0.0486]
a = [0, 0, -0.1104, -0.096, 0, 0]
alpha = [0, np.pi/2, 0, 0, np.pi/2, -np.pi/2]
theta = [0, -np.pi/2, 0, -np.pi/2, np.pi, 0]
robot = RobotArm(d, a, alpha, theta)

# Generate circle points
t = np.arange(0, total_time + sample_rate, sample_rate)
omega = 2 * np.pi / total_time
x = center[0] + radius * np.cos(omega * t)
y = center[1] + radius * np.sin(omega * t)
z = np.full_like(t, center[2])

# Calculate velocities
dx = -radius * omega * np.sin(omega * t)
dy = radius * omega * np.cos(omega * t)
dz = np.zeros_like(t)

# Plot dx, dy, dz vs time
plt.figure(figsize=(10, 6))
plt.plot(t, dx, label='dx')
plt.plot(t, dy, label='dy')
plt.plot(t, dz, label='dz')
plt.title('End Effector Velocity Components vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.savefig('velocity_components_vs_time.png')
plt.close()

# Calculate joint angles and velocities
joint_angles = []
joint_velocities = []

for i in range(len(t)):
    # Current position and velocity
    pos = np.array([x[i], y[i], z[i]])
    vel = np.array([dx[i], dy[i], dz[i], 0, 0, 0])  # Assuming no angular velocity
    
    # Use inverse kinematics to get joint angles (this is a placeholder)
    # In a real scenario, you'd implement inverse kinematics here
    current_angles = np.zeros(6)  # Placeholder
    
    # Calculate joint velocities
    joint_vel = robot.calculate_joint_velocities(vel, current_angles)
    
    joint_angles.append(current_angles)
    joint_velocities.append(joint_vel)

# Convert to numpy arrays for easier manipulation
joint_angles = np.array(joint_angles)
joint_velocities = np.array(joint_velocities)

# Plot joint velocities
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.plot(t, joint_velocities[:, i], label=f'Joint {i+1}')
plt.title('Joint Velocities vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.savefig('joint_velocities_vs_time.png')
plt.close()

print("Plots have been saved as 'velocity_components_vs_time.png' and 'joint_velocities_vs_time.png'")
