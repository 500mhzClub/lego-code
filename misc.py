""" misc pyboard functions, usually linked to the light matrix or other lights """

# Writes {message} to the hub, but it won't wait for 
# that message to complete before starting the next command
def print_text(pyb, message):
    command = f"""
light_matrix.write("{message}")
"""
    pyb.exec(command)
    
    
# Displays {message} on the Hub with await to complete
def print_text_await(pyb, message):
    command = f"""
async def main():
    await light_matrix.write("{message}")

runloop.run(main())
"""
    pyb.exec(command)
    

# flash both lights orange, imitating real hazard warning lights
def hazards(pyb):
    command = """
async def main():
    for i in range(3):
        light.color(light.POWER, color.ORANGE)
        light.color(light.CONNECT, color.ORANGE)
        await runloop.sleep_ms(600)
        
        light.color(light.POWER, color.WHITE)
        light.color(light.CONNECT, color.WHITE)
        await runloop.sleep_ms(600)
        
runloop.run(main())
"""
    pyb.exec(command)
    
    
# flash headlights twice 
def headlight_flash(pyb):
    command = """
async def main():
    running_lights = [30] * 4
    full_beam = [100] * 4
    
    for i in range(2):
        distance_sensor.show(port.C, full_beam)
        await runloop.sleep_ms(300)
        
        distance_sensor.show(port.C, running_lights)
        await runloop.sleep_ms(300)

runloop.run(main())
"""
    pyb.exec(command)
    
    
    
def show_right_arrow(pyb):
    command = """
light_matrix.show_image(29)
"""
    pyb.exec(command)
    

def clear_light_matrix(pyb):
    command = """
light_matrix.clear()
"""
    pyb.exec(command)
    
    
    