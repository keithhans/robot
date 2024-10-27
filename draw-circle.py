import numpy as np
import matplotlib.pyplot as plt
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

# CLIK parameters
K = 10  # Gain for CLIK
dt = sample_rate  # Time step

# Calculate joint angles and velocities
joint_angles = []
joint_velocities = []

q = pin.neutral(model)  # Initial configuration
dq = np.zeros(model.nv)  # Initial velocity

for i in range(len(t)):
    # Current position and velocity
    pos_desired = np.array([x[i], y[i], z[i]])
    vel_desired = np.array([dx[i], dy[i], dz[i]])

    # Compute forward kinematics and Jacobian
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]

    # Current end-effector position
    pos_current = data.oMi[model.njoints-1].translation

    # CLIK
    e = pos_desired - pos_current
    dq = np.linalg.pinv(J) @ (vel_desired + K * e)

    # Integrate
    q = pin.integrate(model, q, dq * dt)

    # Store results
    joint_angles.append(q)
    joint_velocities.append(dq)

# Convert to numpy arrays for easier manipulation
joint_angles = np.array(joint_angles)
joint_velocities = np.array(joint_velocities)

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
actual_x = []
actual_y = []
actual_z = []
for q in joint_angles:
    pin.forwardKinematics(model, data, q)
    pos = data.oMi[model.njoints-1].translation
    actual_x.append(pos[0])
    actual_y.append(pos[1])
    actual_z.append(pos[2])
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
