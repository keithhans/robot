from pymycobot.mycobot import MyCobot
import time

mc = MyCobot("/dev/ttyAMA0", 1000000)

#mc.power_on()


# mode: 0, position; 1, speed
def set_servo_mode(mc, joint_id, mode):
    mc.set_servo_data(joint_id, 33, mode)
    time.sleep(0.1)
    print("mode:", mc.get_servo_data(joint_id, 33))

def get_servo_mode(mc, joint_id):
    return mc.get_servo_data(joint_id, 33)

def set_speed(mc, joint_id, direction, speed):
    print(f"joint {joint_id} at speed {speed}, direction {direction}")
    speed_high = speed // 256
    speed_low = speed % 256
    mask = 1 << 7
    if direction == 1 and joint_id != 3:
        speed_high |= mask
    if direction == 0 and joint_id == 3:
        speed_high |= mask
    start = time.time()
    print("expected:", speed_high, speed_low) 
    mc.set_servo_data(joint_id, 46, speed_low)

    time.sleep(0.01)
    mc.set_servo_data(joint_id, 47, speed_high)
    
    end = time.time()
    print("actual:", mc.get_servo_data(joint_id, 47), 
            mc.get_servo_data(joint_id, 46), 
            "time:", end-start)

def get_speed(mc, joint_id):
    direction = 0
    low = mc.get_servo_data(joint_id, 58)
    if low == None:
        return None
    high = mc.get_servo_data(joint_id, 59)
    if high == None:
        return None
    if(high >= 128):
        high -= 128
        direction = 1
    return (high << 8) + low, direction

def get_encoder(mc, joint_id):
    low = mc.get_servo_data(joint_id, 56)
    if low == None:
        return None
    high = mc.get_servo_data(joint_id, 57)
    if high == None:
        return None
    return (high << 8) + low

def is_running(mc, joint_id):
    return mc.get_servo_data(joint_id, 66) == 1

def stop(mc, joint_id):
    print("stopping...")
    mc.set_servo_data(joint_id, 46, 0)
    time.sleep(0.01)
    mc.set_servo_data(joint_id, 47, 0)
    
    while(is_running(mc, joint_id)):
        print("still running...")
        mc.set_servo_data(joint_id, 46, 0)
        time.sleep(0.01)
        mc.set_servo_data(joint_id, 47, 0)
        time.sleep(0.001)


def test_single_motor(mc, id):

    #set_servo_mode(mc, id, 1)
    print("p ", mc.get_servo_data(id, 37))
    print("i ", mc.get_servo_data(id, 39))
    

    set_speed(mc, id, 0, 25)
    
    last = time.time()

    for i in range(30):
        now = time.time()
        print("encoder:", get_encoder(mc, id), " speed:", get_speed(mc, id), 
                " time ", now - last)
        last = now
    
    stop(mc, id)

    print("encoder:", get_encoder(mc, id))


    print(is_running(mc, id))

    #set_servo_mode(mc, id, 0)
    
    #mc.set_servo_data(id, 40, 1)

def set_target_position(mc, joint_id, target):
    print(f"joint {joint_id} at target {target}")
    target_high = target // 256
    target_low = target % 256
    
    start = time.time()
    print("expected:", target_high, target_low) 
    mc.set_servo_data(joint_id, 42, target_low)

    time.sleep(0.01)
    mc.set_servo_data(joint_id, 43, target_high)
    
    end = time.time()
    time.sleep(0.01)
    print("actual:", mc.get_servo_data(joint_id, 43), 
            mc.get_servo_data(joint_id, 42), 
            "time:", end-start)


def test_position_mode(mc, id, target):

    print("p ", mc.get_servo_data(id, 21))
    print("d ", mc.get_servo_data(id, 22))
    print("i ", mc.get_servo_data(id, 23))
    

    set_speed(mc, id, 0, 25)

    set_target_position(mc, id, target)
    
    last = time.time()

    time.sleep(0.2)

    while is_running(mc, id):
        now = time.time()
        print("encoder:", get_encoder(mc, id), " speed:", get_speed(mc, id), 
                " time ", now - last)
        last = now
    

    print(is_running(mc, id))




def test_multiple_motor(mc):
    set_servo_mode(mc, 5, 1)
    set_servo_mode(mc, 6, 1)

    set_speed(mc, 5, 0, 25)
    set_speed(mc, 6, 0, 50)
    
    #mc.set_servo_data(5, 46, 25)
    #mc.set_servo_data(6, 46, 50)
    #time.sleep(0.01)
    #mc.set_servo_data(5, 47, 128)
    #mc.set_servo_data(6, 47, 128)

    time.sleep(1)
    stop(mc, 5)
    stop(mc, 6)


    set_servo_mode(mc, 5, 0)
    set_servo_mode(mc, 6, 0)


#test_multiple_motor(mc)

#test_single_motor(mc, 1)

test_position_mode(mc, 1, 256 * 8)

