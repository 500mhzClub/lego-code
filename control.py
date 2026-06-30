""" pyboard functions for initialisation and shutdown """
    
# import all Lego hub libaries, initialise lights and orientate light matrix
# imports are listed alphabetically, grouped by whole libraries, then subsections.
def init(pyb):
    command = """
import color
import distance_sensor
import motor
import motor_pair as mp
import runloop

from hub import light
from hub import light_matrix
from hub import motion_sensor
from hub import port

async def main():
    # Setting up the Hub's lights and light_matrix orientation.
    # Relevant colours for us will be ORANGE (8), RED (9) and WHITE (10)
    # Orange for hazards and turn signals, red for brakes and white for running lights
    
    mp.unpair(mp.PAIR_1)
    light.color(light.POWER, color.WHITE)
    light.color(light.CONNECT, color.WHITE)
    running_lights = [30] * 4
    distance_sensor.show(port.C, running_lights)

    # without this line the hub will write 90 degrees wrong
    light_matrix.set_orientation(1)
    motion_sensor.set_yaw_face(motion_sensor.RIGHT)
    
runloop.run(main())
"""
    pyb.exec(command)


# turn all lights to red or off to denote execution complete
def shutdown(pyb):
    command = """
light.color(light.POWER, color.RED)
light.color(light.CONNECT, color.RED)
distance_sensor.clear(port.C)
"""
    pyb.exec(command)


