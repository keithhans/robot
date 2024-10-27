import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pinocchio as pin

# Define the circle parameters
center = np.array([0.1, 0, 0.1])
radius = 0.05
total_time = 10  # seconds
sample_rate = 0.1  # 100ms

# Load the robot model from URDF
urdf_filename = "mycobot_280_pi.urdf"
model = pin.buildModelFromUrdf(urdf_filename)
data = model.createData()

# CLIK parameters
K = 10  # Gain for CLIK
dt = sample_rate  # Time step

# Function to move to a target position using CLIK
def move_to_target(q_init, target_position, max_iterations=1000):
    q = q_init.copy()
    for _ in range(max_iterations):
        pin.forwardKinematics(model, data, q)
        current_position = data.oMi[model.njoints-1].translation
        error = target_position - current_position
        if np.linalg.norm(error) < 1e-4:  # Convergence threshold
            break
        J = pin.computeJointJacobians(model, data, q)
        J = J[:3, :]  # We only need the translational part
        dq = np.linalg.pinv(J) @ (K * error)
        q = pin.integrate(model, q, dq * dt)
    return q

# Move to the start position of the circle
start_position = center + np.array([radius, 0, 0])
q_neutral = pin.neutral(model)
q_start = move_to_target(q_neutral, start_position)

print("Moved to start position.")
pin.forwardKinematics(model, data, q_start)
print(f"Start position: {data.oMi[model.njoints-1].translation}")

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

# Calculate joint angles and velocities
joint_angles = [q_start]
joint_velocities = []
actual_x, actual_y, actual_z = [], [], []

q = q_start
for i in range(len(t)):
    pos_desired = np.array([x[i], y[i], z[i]])
    vel_desired = np.array([dx[i], dy[i], dz[i]])

    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]

    pos_current = data.oMi[model.njoints-1].translation
    actual_x.append(pos_current[0])
    actual_y.append(pos_current[1])
    actual_z.append(pos_current[2])

    e = pos_desired - pos_current
    dq = np.linalg.pinv(J) @ (vel_desired + K * e)

    q = pin.integrate(model, q, dq * dt)

    joint_angles.append(q)
    joint_velocities.append(dq)

# Convert to numpy arrays
joint_angles = np.array(joint_angles)
joint_velocities = np.array(joint_velocities)

# Plot x, y, z vs time
plt.figure(figsize=(10, 6))
plt.plot(t, actual_x, label='x')
plt.plot(t, actual_y, label='y')
plt.plot(t, actual_z, label='z')
plt.title('End Effector Position Components vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('position_components_vs_time.png')
plt.close()

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

# Plot joint velocities
plt.figure(figsize=(12, 8))
for i in range(model.nv):
    plt.plot(t, joint_velocities[:, i], label=f'Joint {i+1}')
plt.title('Joint Velocities vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.savefig('joint_velocities_vs_time.png')
plt.close()

print("Plots have been saved as 'velocity_components_vs_time.png' and 'joint_velocities_vs_time.png'")

# Verify final position
pin.forwardKinematics(model, data, joint_angles[-1])
final_position = data.oMi[model.njoints-1].translation
print(f"Final end-effector position: {final_position}")
print(f"Desired final position: {pos_desired}")
print(f"Position error: {np.linalg.norm(final_position - pos_desired)}")

# Plot trajectory
plt.figure(figsize=(10, 10))
plt.plot(x, y, label='Desired')
plt.plot(actual_x, actual_y, label='Actual')
plt.title('End Effector Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.savefig('trajectory.png')
plt.close()

print("Trajectory plot has been saved as 'trajectory.png'")

