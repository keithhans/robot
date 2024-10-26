import numpy as np
import time

class RobotArm:
    def __init__(self, 
                 d = [0.13156, 0, 0, 0.0634, 0.07505, 0.0456], 
                 a = [0, 0, -0.1104, -0.096, 0, 0], 
                 alpha = [0, np.pi/2, 0, 0, np.pi/2, -np.pi/2], 
                 theta = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]):
        self.d = d  # Link offsets
        self.a = a  # Link lengths
        self.alpha = alpha  # Link twists
        self.theta = theta  # Joint angles (initial values)

    def dh_matrix(self, d, theta, a, alpha):
        """Calculate the DH transformation matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), d*np.cos(alpha)],
            [0, 0, 0, 1]
        ])

    def calculate_jacobian(self, theta):
        """Calculate the Jacobian matrix."""
        T = [np.eye(4)]
        for i in range(6):
            T.append(T[i] @ self.dh_matrix(self.d[i], self.theta[i] + theta[i], self.a[i], self.alpha[i]))
        
        J = np.zeros((6, 6))
        for i in range(6):
            z = T[i][:3, 2]
            p = T[6][:3, 3] - T[i][:3, 3]
            J[:3, i] = np.cross(z, p)
            J[3:, i] = z
        
        return J

    def calculate_joint_velocities(self, end_effector_velocity, theta):
        """Calculate joint velocities given end-effector velocity."""
        J = self.calculate_jacobian(theta)
        J_pseudo_inv = np.linalg.pinv(J)
        joint_velocities = J_pseudo_inv @ end_effector_velocity
        return joint_velocities

    def calculate_end_effector_position(self, theta):
        """Calculate the end-effector position given joint angles."""
        T = np.eye(4)
        for i in range(6):
            T = T @ self.dh_matrix(self.d[i], self.theta[i] + theta[i], self.a[i], self.alpha[i])
        return T[:3, 3]  # Return the position (x, y, z)

    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def calculate_forward_kinematics(self, theta):
        """Calculate the position and orientation of each joint frame."""
        T = np.eye(4)
        positions = []
        orientations = []
        for i in range(6):
            T = T @ self.dh_matrix(self.d[i], self.theta[i] + theta[i], self.a[i], self.alpha[i])
            positions.append(T[:3, 3])
            orientations.append(self.rotation_matrix_to_euler_angles(T[:3, :3]))
        return positions, orientations

# Example usage
if __name__ == "__main__":
    # Define D-H parameters
    #d = [0.13156, 0, 0, 0.0634, 0.07505, 0.0456]  # Link offsets
    #a = [0, 0, -0.1104, -0.096, 0, 0]  # Link lengths
    #alpha = [0, np.pi/2, 0, 0, np.pi/2, -np.pi/2]  # Link twists
    #theta = [0, -np.pi/2, 0, -np.pi/2, np.pi/2, 0]  # Joint angles (initial values)

    # Create RobotArm instance
    # robot = RobotArm(d, a, alpha, theta)
    robot = RobotArm()

    # Example usage
    initial_joint_angles = [0, 0, 0, 0, 0, 0]  # Use the initial theta values
    current_joint_angles = [0, np.pi/4, -np.pi/6, np.pi/3, -np.pi/4, np.pi/6]
    end_effector_vel = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])  # [vx, vy, vz, wx, wy, wz]

    # Calculate and print the end-effector position at initial angles
    start_time = time.time()
    initial_position = robot.calculate_end_effector_position(initial_joint_angles)
    end_time = time.time()
    position_calculation_time = end_time - start_time

    print("End-effector position at initial angles:")
    print(f"X: {initial_position[0]:.4f}, Y: {initial_position[1]:.4f}, Z: {initial_position[2]:.4f}")
    print(f"Time taken to calculate end-effector position: {position_calculation_time:.6f} seconds")

    # Calculate and print joint velocities
    start_time = time.time()
    joint_velocities = robot.calculate_joint_velocities(end_effector_vel, current_joint_angles)
    end_time = time.time()
    velocity_calculation_time = end_time - start_time

    print("\nJoint velocities:")
    for i, vel in enumerate(joint_velocities):
        print(f"Joint {i+1}: {vel:.4f} rad/s")
    print(f"Time taken to calculate joint velocities: {velocity_calculation_time:.6f} seconds")

    # Calculate and print positions and orientations of each joint frame at initial angles
    joint_positions, joint_orientations = robot.calculate_forward_kinematics(initial_joint_angles)
    print("\nJoint frame positions and orientations at initial configuration:")
    for i, (pos, orient) in enumerate(zip(joint_positions, joint_orientations)):
        print(f"Joint {i+1}:")
        print(f"  Position: X: {pos[0]:.4f}, Y: {pos[1]:.4f}, Z: {pos[2]:.4f}")
        print(f"  Orientation (Euler angles in radians): Roll: {orient[0]:.4f}, Pitch: {orient[1]:.4f}, Yaw: {orient[2]:.4f}")
