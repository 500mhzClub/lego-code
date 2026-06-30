"""
Serves as a combination of software Categories 1 and 2, or control and object detection.
The functions for these two categories can be found in driver() and detector() respectively, in streams.py.
Once all modules have been imported, all variables defined, and the videostream and hub initalised,
the functions are defined into separate threads and executed in parallel.
The program can always be completely shutdown by pressing Ctrl C twice - this will also halt the robot.

(7) .py files required for Autonomous Lego are:
AutoLego.py      - control of initialisation, threading & shutdown
control.py       - pyboard functions for initialisation and shutdown of the robot
misc.py          - miscellaneous pyboard functions
movement.py      - pyboard functions for moving the robot
pyboard.py       - the Pyboard class
streams.py       - the functions driver() and detector() as mentioned above
videostream.py   - the VideoStream class

Additional dependencies are:
the tfl-env folder to provide all modules required below, and any subsequent ones imported in the above .py files
the tfl_model folder, which must contain:
    edgetpu.tflite
    labelmap.txt
"""

print("    Welcome to Autonomous Lego.")
from datetime import datetime
start_time = datetime.now()
print()
print(" Initialising camera stream and robot...")
print()

# Required libraries and classes, sorted alphabetically
import control
import misc
import movement
import numpy as np
import os
from pyboard import Pyboard
import queue
from streams import driver, detector
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import threading
import time
from videostream import VideoStream

"""GLOBAL CONSTANTS"""

CWD_PATH = os.getcwd()            # Get path to current working directory - /home/pi/Desktop/TGR
device   = "/dev/ttyACM0"         # for pyboard
imW, imH = 1280, 720              # the dimensions of the VideoStream images read from the camera
GRAPH_NAME    = "edgetpu.tflite"  # name of TFL model, compiled for Edge TPU 
LABELMAP_NAME = "labelmap.txt"    # name of line separated class names text file
MODEL_NAME    = "tfl_model"       # path of TFL model folder which contains the above 2 files

# Full path to .tflite file, which contains the model that is used for object detection
PATH_TO_MODEL = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)     # CWD/tfl_model/edgetpu.tflite
# Full path to label map file, which contains line spaced class names
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME, LABELMAP_NAME)  # CWD/tfl_model/labelmap.txt



"""Init labels"""
with open(PATH_TO_LABELS, "r") as f:
    labels = [line.strip() for line in f.readlines()]
f.close()
# labels is now just an array of all possible objects the model is trained on



"""Initialise interpreter and model details"""
interpreter = Interpreter(model_path = PATH_TO_MODEL,
                          experimental_delegates = [load_delegate("libedgetpu.so.1.0")])

interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]["shape"][1] # 300
width  = input_details[0]["shape"][2] # 300
# height and width are the image dimensions expected by the model



"""Initialise video stream"""
videostream = VideoStream(resolution = (imW,imH), framerate = 30).start()
time.sleep(0.3)



"""Initialise robot"""
pyb = Pyboard(device)
pyb.enter_raw_repl()
control.init(pyb)
misc.print_text_await(pyb, "Hello.")
misc.headlight_flash(pyb)



"""End of initialisation"""

end_time = datetime.now()
init_time = end_time - start_time
init_time_seconds = float(f"{init_time.total_seconds():.3g}")
print(f" Initialisation completed successfully in {init_time_seconds} seconds.")
time.sleep(0.3)
print(" Running Autonomous Lego...")
print()
time.sleep(1.5)



"""Autonomous Lego"""

"""Initialise queue, event and threads"""
q = queue.Queue()
object_detected = threading.Event()
shutdown_event  = threading.Event()

drive_thread = threading.Thread(target = driver, args = (object_detected, shutdown_event, q, pyb, ))
# So that we can make this main file tidy by defining the detector function in another file, 
# we have to pass a number of arguments to it.
# most variables/constants requiring computation are passed, all others are defined within the functions in streams.py
detect_thread = threading.Thread(target = detector, args = (object_detected, shutdown_event, pyb, videostream,
                                                            width, height, interpreter, input_details, output_details,
                                                            imW, imH, labels, q, ))

# at these lines, both Categories 1 and 2 execute
drive_thread.start()
detect_thread.start()

try:
    while True:
        time.sleep(1)
        pass
except KeyboardInterrupt:
    """Cleanup and Finish"""
    # We have to Ctrl C twice to shut it down
    shutdown_event.set()
    movement.stop(pyb)
    control.shutdown(pyb)
    pyb.exit_raw_repl()
    pyb.close()
    videostream.stop()
    
print()
print("Completed with 0 errors.")
print()
