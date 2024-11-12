import pinocchio as pin
import numpy as np
from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyAMA0", 1000000)
mc.set_fresh_mode(0)


# Load the robot model from URDF
urdf_filename = "mycobot_280_pi.urdf"
model = pin.buildModelFromUrdf(urdf_filename)
data = model.createData()

JOINT_ID = 6


def fk(q):
    pin.forwardKinematics(model, data, q)

    pos_current = data.oMi[JOINT_ID].translation * 1000
    rot_current = data.oMi[JOINT_ID].rotation
    rpy_current = pin.rpy.matrixToRpy(rot_current) / np.pi * 180

    return [pos_current[0], pos_current[1], pos_current[2],
            rpy_current[0], rpy_current[1], rpy_current[2]]

q1 = np.array([0, 0, -np.pi/2, 0, 0, 0])
c1 = fk(q1)
print(c1)


q2 = np.array([0, 0, 0, 0, 0, 0])
c2 = fk(q2)

print(c2)

q3 = np.array([0.139, -0.512, -2.25, 1.19, 0.003, 0.146])
c3 = fk(q3)
print(c3)

