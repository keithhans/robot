import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pinocchio as pin

# Define the circle parameters
center = np.array([0.18, 0, 0.1])
radius = 0.05
total_time = 10  # seconds
sample_rate = 0.1  # 100ms

# Load the robot model from URDF
urdf_filename = "mycobot_280_pi.urdf"
model = pin.buildModelFromUrdf(urdf_filename)
data = model.createData()

JOINT_ID = 6
eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

def move_to_target(q_init, target_position, target_rotation=None):
    if target_rotation is None:
        target_rotation = pin.utils.rpyToMatrix(-3.1416, 0, -1.5708)  # Default rotation if not specified
    oMdes = pin.SE3(target_rotation, target_position)
    q = q_init.copy()
    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pin.log(iMd).vector
        if np.linalg.norm(err) < eps:
            return np.array(q), True  # Converged successfully
        if i >= IT_MAX:
            print(f"Warning: max iterations reached without convergence. error norm:{np.linalg.norm(err)}")
            return np.array(q), False  # Did not converge
        J = pin.computeJointJacobian(model, data, q, JOINT_ID)
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v * DT)
        i += 1

# Move to the start position of the circle
start_position = center + np.array([radius, 0, 0])
q_neutral = pin.neutral(model)
q_start, _ = move_to_target(q_neutral, start_position)

print("Moved to start position.")
pin.forwardKinematics(model, data, q_start)
print(f"Start position: {data.oMi[JOINT_ID].translation}")

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
joint_angles = []
joint_velocities = []
actual_x, actual_y, actual_z = [], [], []
actual_roll, actual_pitch, actual_yaw = [], [], []

q = q_start
initial_rotation = data.oMi[JOINT_ID].rotation  # Get initial rotation

for i in range(len(t)):
    pos_desired = np.array([x[i], y[i], z[i]])
    
    # Use move_to_target to get joint angles, maintaining initial orientation
    q, converged = move_to_target(q, pos_desired, initial_rotation)
    
    if not converged:
        print(f"Skipping point at t = {t[i]} pos = {pos_desired}")
        continue
    
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    J = pin.getJointJacobian(model, data, JOINT_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    pos_current = data.oMi[JOINT_ID].translation
    rot_current = data.oMi[JOINT_ID].rotation
    rpy_current = pin.rpy.matrixToRpy(rot_current)

    actual_x.append(pos_current[0])
    actual_y.append(pos_current[1])
    actual_z.append(pos_current[2])
    actual_roll.append(rpy_current[0])
    actual_pitch.append(rpy_current[1])
    actual_yaw.append(rpy_current[2])

    vel_desired = np.array([dx[i], dy[i], dz[i], 0, 0, 0])  # Assuming no angular velocity
    dq = np.linalg.pinv(J) @ vel_desired

    joint_angles.append(q)
    joint_velocities.append(dq)

# Convert to numpy arrays
joint_angles = np.array(joint_angles)
joint_velocities = np.array(joint_velocities)

# Calculate actual trajectory
actual_x, actual_y, actual_z = [], [], []
actual_roll, actual_pitch, actual_yaw = [], [], []
for q in joint_angles:
    pin.forwardKinematics(model, data, q)
    pos = data.oMi[JOINT_ID].translation
    actual_x.append(pos[0])
    actual_y.append(pos[1])
    actual_z.append(pos[2])
    
    rot_current = data.oMi[JOINT_ID].rotation
    rpy_current = pin.rpy.matrixToRpy(rot_current)

    actual_roll.append(rpy_current[0])
    actual_pitch.append(rpy_current[1])
    actual_yaw.append(rpy_current[2])

# Create a new time array based on the actual number of points
t_actual = np.linspace(0, total_time, len(actual_x))

# Plot x, y, z vs time
plt.figure(figsize=(10, 6))
plt.plot(t_actual, actual_x, label='x')
plt.plot(t_actual, actual_y, label='y')
plt.plot(t_actual, actual_z, label='z')
plt.title('End Effector Position Components vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.savefig('position_components_vs_time.png')
plt.close()

# Plot dx, dy, dz vs time (use original t for desired velocities)
plt.figure(figsize=(10, 6))
plt.plot(t, dx, label='dx (desired)')
plt.plot(t, dy, label='dy (desired)')
plt.plot(t, dz, label='dz (desired)')
plt.title('Desired End Effector Velocity Components vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.savefig('velocity_components_vs_time.png')
plt.close()

# Plot joint velocities
plt.figure(figsize=(12, 8))
for i in range(model.nv):
    plt.plot(t_actual, joint_velocities[:, i], label=f'Joint {i+1}')
plt.title('Joint Velocities vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)
plt.savefig('joint_velocities_vs_time.png')
plt.close()

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

# Plot roll, pitch, yaw vs time
plt.figure(figsize=(10, 6))
plt.plot(t_actual, actual_roll, label='Roll')
plt.plot(t_actual, actual_pitch, label='Pitch')
plt.plot(t_actual, actual_yaw, label='Yaw')
plt.title('End Effector Orientation (RPY) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid(True)
plt.savefig('orientation_rpy_vs_time.png')
plt.close()

print("All plots have been saved, including the new orientation (RPY) plot.")

# Verify final position
pin.forwardKinematics(model, data, joint_angles[-1])
final_position = data.oMi[JOINT_ID].translation
print(f"Final end-effector position: {final_position}")
print(f"Desired final position: {pos_desired}")
print(f"Position error: {np.linalg.norm(final_position - pos_desired)}")
