""" pyboard functions concerning movement """


# run_for_degrees takes (a port, the number of degrees to turn for, and the velocity)

# we split "turn_left" into the movement and the signal
# the signal will flash the left indicator and display a left arrow
# then run both functions asyncronously
def turn_left_motor(pyb):
    command = """
async def left():
    # port, degrees, velocity
    await motor.run_for_degrees(port.E, 420, 300)
    
async def signal():
    for i in range(5):
        light.color(light.CONNECT, color.ORANGE)
        light_matrix.show_image(33)
        await runloop.sleep_ms(420)
        
        light.color(light.CONNECT, color.WHITE)
        light_matrix.clear()
        await runloop.sleep_ms(420)
    
runloop.run(left(), signal())
"""
    pyb.exec(command)

# 0-900 is 1/4 left
def turn_left_gyro(pyb):
    command = """
async def left():
    while motion_sensor.tilt_angles()[0] < 900:
        mp.move_tank(mp.PAIR_1, -100, 100)
    mp.stop(mp.PAIR_1)

mp.pair(mp.PAIR_1, port.A, port.E)
motion_sensor.reset_yaw(0)
runloop.run(left())
mp.unpair(mp.PAIR_1)
"""
    pyb.exec(command)

# for notes, see turn_left_motor
def turn_right_motor(pyb):
    command = """
async def right():
    await motor.run_for_degrees(port.A, -420, 300)
    
async def signal():
    for i in range(5):
        light.color(light.POWER, color.ORANGE)
        light_matrix.show_image(29)
        await runloop.sleep_ms(420)
        
        light.color(light.POWER, color.WHITE)
        light_matrix.clear()
        await runloop.sleep_ms(420)
    
runloop.run(right(), signal())
"""
    pyb.exec(command)

# once the yaw has been reset to 0, 900 will be a half turn left, and -900 a half turn right
def turn_right_gyro(pyb):
    command = """
light_matrix.show_image(29)
async def left():
    while motion_sensor.tilt_angles()[0] > -900:
        mp.move_tank(mp.PAIR_1, 150, -150)
    mp.stop(mp.PAIR_1)

mp.pair(mp.PAIR_1, port.A, port.E)
motion_sensor.reset_yaw(0)
runloop.run(left())
mp.unpair(mp.PAIR_1)
light_matrix.clear()
"""
    pyb.exec(command)



# this will drive the robot forward for a specificied number of ms and velocity
# we unpair and pair PAIR_1 as our wheels
# mp.move_for_time takes (a mp.PAIR, number of ms, steering (0 for straight)
# a negative velocity value will reverse the bot
def drive_forward(pyb):
    command = """
def main(): 
    mp.move_for_time(mp.PAIR_1, 1000, 0, velocity = 660, stop = motor.BRAKE)

main()
    
"""
    pyb.exec(command)
    
    

# from the SPIKE Prime docs:
# "If the distance sensor cannot read a valid distance, it will return -1."
# thus the below code basically says - "if the reading is an error, proceed anyway"
def drive_until_wall(pyb):
    command = """
def main():
    mp.pair(mp.PAIR_1, port.A, port.E) 
    while True:
        if (distance_sensor.distance(port.C) > 150) or (distance_sensor.distance(port.C) == -1):
            mp.move(mp.PAIR_1, 0, velocity = 1110)
        else:
            mp.stop(mp.PAIR_1)
            break
    mp.stop(mp.PAIR_1)
    
main()
"""
    pyb.exec(command)

# things get a bit tricky here - basically there is no way to "reuse" code from another function without just
# writing it out again, because the code with a command has no awareness of top level functions
# it does, however, have awareness of functions within that exact command, so we can just copy paste the
# function we want, and everything will work, albeit with a few repeated lines
# for i in range, or while True:
#     display current number iteration
#     drive_until_wall (code copied)
#     turn_right_motor (code copied)

def drive_until_wall_loop(pyb):
    command = """
async def right():
    await motor.run_for_degrees(port.A, -420, 200)
    
async def signal():
    for i in range(3):
        light.color(light.POWER, color.ORANGE)
        light_matrix.show_image(29)
        await runloop.sleep_ms(320)
        
        light.color(light.POWER, color.WHITE)
        light_matrix.clear()
        await runloop.sleep_ms(320)

async def main():
    mp.pair(mp.PAIR_1, port.A, port.E)
    for i in range(1,8):
        iteration = str(i)
        await light_matrix.write(f"{iteration}")
        
        while True:
            if (distance_sensor.distance(port.C) > 400) or (distance_sensor.distance(port.C) == -1):
                mp.move(mp.PAIR_1, 0, velocity = 300)
            else:
                mp.stop(mp.PAIR_1)
                break
        mp.stop(mp.PAIR_1)
        
        runloop.run(right(), signal())
        
runloop.run(main())
"""
    pyb.exec(command)
    
    
    
    
def stop(pyb):
    command = """
mp.pair(mp.PAIR_1, port.A, port.E) 
mp.stop(mp.PAIR_1)
mp.unpair(mp.PAIR_1)
"""
    pyb.exec(command)
    
def stop_with_hazards(pyb):
    command = """
light_matrix.show_image(1)
async def stop():
    mp.pair(mp.PAIR_1, port.A, port.E) 
    mp.stop(mp.PAIR_1)
    mp.unpair(mp.PAIR_1)
    
async def hazards():
    for i in range(6):
        light.color(light.POWER, color.ORANGE)
        light.color(light.CONNECT, color.ORANGE)
        await runloop.sleep_ms(1000)
        
        light.color(light.POWER, color.WHITE)
        light.color(light.CONNECT, color.WHITE)
        await runloop.sleep_ms(1000)
        
runloop.run(stop(), hazards())
"""
    pyb.exec(command)
    
    
def drive(pyb):
    command = """
mp.pair(mp.PAIR_1, port.A, port.E)
mp.move(mp.PAIR_1, 0, velocity = 300)
mp.unpair(mp.PAIR_1)
"""
    pyb.exec(command)
    
    
def drive_slow(pyb):
    command = """
mp.pair(mp.PAIR_1, port.A, port.E)
mp.move(mp.PAIR_1, 0, velocity = 150)
mp.unpair(mp.PAIR_1)
"""
    pyb.exec(command)
    
    
def drive_fast(pyb):
    command = """
mp.pair(mp.PAIR_1, port.A, port.E)
mp.move(mp.PAIR_1, 0, velocity = 700)
mp.unpair(mp.PAIR_1)
"""
    pyb.exec(command)   
    
    
def overtake_right(pyb):
    command = """
async def main():
    mp.pair(mp.PAIR_1, port.A, port.E)
    
    # turn slight right
    motion_sensor.reset_yaw(0)
    while motion_sensor.tilt_angles()[0] > -450: # 1/4 turn right
        mp.move_tank(mp.PAIR_1, 300, -300)
    mp.stop(mp.PAIR_1)

    
    # simulate moving out
    await mp.move_for_time(mp.PAIR_1, 1200, 0, velocity = 700)
    
    # return to og angle
    motion_sensor.reset_yaw(0)
    while motion_sensor.tilt_angles()[0] < 400: # 1/4 turn left
        mp.move_tank(mp.PAIR_1, -300, 300)
    mp.stop(mp.PAIR_1)
    
    # simulate overtaking
    mp.move(mp.PAIR_1, 0, velocity = 700)
    
    mp.unpair(mp.PAIR_1)
    
runloop.run(main())
"""
    pyb.exec(command)


