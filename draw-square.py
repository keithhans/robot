from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyAMA0", 1000000)

mc.set_fresh_mode(0)


speed = 100
x = 60
y = -20
z = 170
t = 3
mode = 1

for count in range(2):
  mc.send_coords([x+190.4, y-63.0, z, 179.2, -0.26, -89.82],speed,mode)
  time.sleep(t)
  
  mc.send_coords([x+100.4, y-63.0, z, 179.2, -0.26, -89.82],speed,mode)
  time.sleep(t)
  
  mc.send_coords([x+100.4, y-2.0, z, 179.2, -0.26, -89.82],speed,mode)
  time.sleep(t)
  
  mc.send_coords([x+190.4, y-2.0, z, 179.2, -0.26, -89.82],speed,mode)
  time.sleep(t)


