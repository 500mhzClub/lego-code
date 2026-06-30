from collections import Counter
import cv2
from datetime import datetime
import misc
import movement
import numpy as np
import time


UNKNOWN_OBJECT = "__unknown_object__"
DISTANCE_SENSOR_STOP_THRESHOLD_MM = 400
UNKNOWN_OBJECT_DISTANCE_THRESHOLD_MM = DISTANCE_SENSOR_STOP_THRESHOLD_MM


def _stop_if_obstacle_is_close(pyb, shutdown_event, label="obstacle"):
    try:
        distance_mm = movement.read_distance_mm(pyb)
    except ValueError as exc:
        print(f"Distance sensor read failed: {exc}")
        return False

    if distance_mm != -1 and distance_mm <= DISTANCE_SENSOR_STOP_THRESHOLD_MM:
        print(f"{label} detected at {distance_mm}mm")
        shutdown_event.set()
        movement.stop_with_hazards(pyb)
        return True

    return False


""" Category 1 (C1) - Movement & Control """
def driver(object_detected, shutdown_event, q, pyb):
    # tells the robot to drive straight indefinitely
    movement.drive(pyb)
    while True:
        if shutdown_event.is_set():
            movement.stop(pyb)
            break

        else:
            if object_detected.is_set():
                # object detected - react, clear the flags and continue
                detection_event = q.get()
                if isinstance(detection_event, dict):
                    class_detected = detection_event.get("class")
                    detected_label = detection_event.get("label", class_detected)
                else:
                    class_detected = detection_event
                    detected_label = detection_event
                # enact differing reaction based on the class detected here, which we recieve from the queue
                
                if   class_detected == "stop sign":
                    shutdown_event.set() # this will irreversibly halt execution
                    movement.stop_with_hazards(pyb)
                    
                elif class_detected == "traffic light":
                    movement.stop(pyb)
                    time.sleep(3) # simulate waiting for the light to turn green 
                    movement.turn_right_gyro(pyb)
                    
                elif class_detected == "person":
                    misc.headlight_flash(pyb)
                    movement.drive_slow(pyb) # simulate driving past slowly
                    time.sleep(6) # how long would we like the decreased speed to last
                    
                elif class_detected == "bicycle":
                    movement.overtake_right(pyb) # simulate pulling out and overtaking
                    time.sleep(4) # how long would we like the increased speed to last

                elif class_detected == "cup":
                    print("CUP DETECTED! Driving through it.")
                    movement.drive_fast(pyb)
                    time.sleep(1.5)

                elif class_detected == "new class":
                    # your new function call here
                    pass

                elif class_detected == UNKNOWN_OBJECT:
                    if not _stop_if_obstacle_is_close(pyb, shutdown_event, f"UNKNOWN OBJECT DETECTED: {detected_label}"):
                        print(f"UNKNOWN OBJECT DETECTED: {detected_label}, but distance sensor is clear")
                    
                else:
                    print(f"UNKNOWN OBJECTED DETECTED: {class_detected}")
                    pass

                    
                # regardless of the class, clear the flags and continue
                q.task_done()
                object_detected.clear()
                if shutdown_event.is_set():
                    break
                movement.drive(pyb)
            else:
                # no object detected
                if _stop_if_obstacle_is_close(pyb, shutdown_event):
                    break
                time.sleep(0.08) # approx matches the rate at which the detector() while loop fires
                # this just helps with frames processed per second (fpps), otherwise the while True fires without delay
                pass


    
    movement.stop(pyb)
    print("Shutting Down...")
    raise KeyboardInterrupt
    """End of C1"""
    


""" Category 2 (C2) - Perception, Detection & Reaction"""
# in an attempt to clean up both this file and AutoLego.py, we define any trivial variables here, 
# and most others which require some kind of computation, in AutoLego.py
# while this does make the definition a little messy with arguments, 
# it makes the rest of both files alot cleaner
# the definitions of all parameters can be found in AutoLego.py, where detector() is called from 
def detector(object_detected, shutdown_event, pyb, videostream, width, height, interpreter, 
             input_details, output_details, imW, imH, labels, q):
    iter_count = 1               # purely for logging
    x = 13                       # the number of frames stored in memory before deleting the oldest      """13""" 
    object_data_stream = []      # stores the last x frames in memory
    min_conf_threshold = 0.3     # the minimum score for an object to be recorded                        """0.3""" 
    min_score_threshold = 40     # the minimum score for an object to be deemed significant              """40""" 
    min_size_threshold = 42000   # the minimum size for an object to be deemed close                     """42000""" 
    boxes_idx, classes_idx, scores_idx = 0, 1, 2    # the respective indexes within output_details
    # the only classes we will react to - this is to avoid reacting to false positives of other classes
    implemented_classes = ["stop sign", "traffic light", "person", "bicycle", "cup"]
    
    
    while not shutdown_event.is_set():
        while True:
            start_time = datetime.now()
            
            """Perception"""
            # Grab frame from video stream, convert it to RGB and resize to expected shape [1 x height x width x 3]
            frame = videostream.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height)) # to 300 x 300
            input_data    = np.expand_dims(frame_resized, axis = 0)
            
            """Detection"""
            # Perform the inference by running the model with the image as input
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            # Retrieve detection results
            # defines all boxes, classes and scores of all "objects" the model has detected in that frame, as separate variables
            boxes   = interpreter.get_tensor(output_details[boxes_idx]["index"])[0]    # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]["index"])[0]  # Class indexes of detected objects
            scores  = interpreter.get_tensor(output_details[scores_idx]["index"])[0]   # Confidence scores of detected objects
            
            if len(scores) == 0: # if no object was detected in this frame
                # we still append because we need some way of knowing this was a dud frame
                object_data_stream.append(["None", 0, 0])
                
            else: # at least 1 object was detected in this frame
                max_score_index = np.argmax(scores)  # index of highest scoring object
                # will return the first if there are multiple
                if (scores[max_score_index] < min_conf_threshold): # if even the highest score is insignificant:
                    # without this catching if statement, we will "miss" frames which had >0 objects
                    # but all objects were insignificant
                    object_data_stream.append(["None", 0, 0])
                
                else:
                    i = max_score_index # purely to improve readability
                    # the score is now the highest of that frame, and is significant
                    # thus we append something other than None to our data stream
                    
                    # Get bounding box coordinates
                    # Interpreter can return coordinates that are outside of image dimensions,
                    # thus we need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    # calculate the area of the box - this is to determine how close the object is
                    box_base   = xmax - xmin
                    box_height = ymax - ymin
                    box_area   = box_base * box_height
                    # if an object could somehow occupy the entire camera's FOV, it would have an area of 921,600 (1280 x 720)
                    # our size threshold is currently set at 42,000, or ~20% of this
                    
                    label_index = int(classes[i])
                    try:
                        object_name = labels[label_index] # Look up object name from "labels" array using class index
                    except IndexError:
                        # kept getting index errors while accessing the label map so added this to help debugging
                        print("INDEX ERROR while parsing label. Searched for index:", label_index, "but length of labels is", len(labels))
                        raise IndexError
                    
                    # create a 1x3 list of [name, area, size (as a %)] for the highest scoring object in this frame
                    object_data = [str(object_name), int(box_area), int(scores[i] * 100)]
                    object_data_stream.append(object_data) # x*3 matrix
                    
            # this if clause needs to run in all eventualities, whether None or an object was appended
            # this is to fix an earlier bug where len(object_data_stream) grew much, much larger than intended
            if (len(object_data_stream) > x): 
                del(object_data_stream[0])    # delete the oldest frame
                    
                """End of appending to object_data_stream"""
                    
                    
            """Reaction"""
            
            """
            Excluding the initial few frames, object_data_stream should now be a x*3 matrix of the last x frames,
            with each 1*3 vector of the form [label, size, score] for the highest scoring object in that frame.
            If no objects were detected in that frame, OR the higher scoring object was below the significance level
            to be recorded, then ["None", 0, 0] was appended - this is the same format as [label, size, score],
            so will not cause any errors from transposing. 
            We must transpose to a different variable as tranposing object_data_stream would cause errors on the next iteration
            """
            
            # list zip is one of many ways to transpose a 2 dimensional list in python
            object_data_stream_T = list(zip(*object_data_stream)) # transposed - now a 3*x matrix
            # transposing will yield 3 vectors of length x: [labels, sizes, scores]
            # we define these as independent arrays for simplicty
            x_labels = object_data_stream_T[0]
            x_sizes  = object_data_stream_T[1] 
            x_scores = object_data_stream_T[2]
            
            # find the mode label in the last x frames
            # statistics.mode() crashed in the case of two equal modes, and statistics.multimode() is
            # not available in this old Python version, thus we had to implement a Counter bug fix
            label_counts = Counter(x_labels)
            max_count    = max(label_counts.values())
            mode_object_label = [k for k, v in label_counts.items() if v == max_count][-1] # -1 takes the most recent mode
            
            # if no object was significantly, consistently detected in the last x frames:
            # (we also had to add an UNKNOWN workaround to avoid index errors when accessing labels - labelmap.txt is 
            # now packed with a few "UNKNOWN" labels at the end to catch these errors)
            # we also check if the mode object is in our implemented_classes list,
            # as we only want to react to objects we have defined reactions for
            # this is to fix an earlier bug where the model would misclassify objects, and this would flood the terminal
            if (mode_object_label == "None") or (mode_object_label == "UNKNOWN"): 
                pass # skip all the reaction
            else:
                """an object of interest is the mode"""
                # find the indicies at which the mode object was the highest scoring in the last x frames, as we require a count of this 
                mode_indicies = [j for j, v in enumerate(x_labels) if v == mode_object_label]
                num_mode_occurences = len(mode_indicies)
                
                # then create a list of those respective sizes...
                mode_object_sizes = []
                for k in range(len(x_sizes)):
                    if k in mode_indicies:
                        mode_object_sizes.append(x_sizes[k])
                      
                # ...and scores      
                mode_object_scores = []
                for k in range(len(x_scores)):
                    if k in mode_indicies:
                        mode_object_scores.append(x_scores[k])
            
                # finally, sum those lists
                """
                Explicitly - these two numbers are the sums of respective sizes and scores for the 
                single most commonly occuring significant object in the last x frames. The class is also of interest.
                We can hopefully use these 3 numbers to determine if an object is significant enough to react to, or not:

                x tells us for how long this object has been significant (currently around 1 second)
                size_sum tells us if the object is close enough - the closer the object, the bigger the size
                score_sum tells us if the model is confident in its classification - note that confidence
                classifications are already restricted to above a certain threshold in the above for loop
                """
                mode_object_size_sum  = sum(mode_object_sizes)
                mode_object_score_sum = sum(mode_object_scores)
                # we create a dynamic threshold based upon the number of times the object occurred in the last x frames
                size_threshold  = num_mode_occurences * min_size_threshold
                score_threshold = num_mode_occurences * min_score_threshold

                # this is set to never fire unless the flag has been reset since the last object
                if (mode_object_size_sum > size_threshold) and (mode_object_score_sum > score_threshold) and (not object_detected.is_set()):
                    if mode_object_label in implemented_classes:
                        q.put(mode_object_label) # append the label to the queue
                    else:
                        q.put({"class": UNKNOWN_OBJECT, "label": mode_object_label})

                    print(f"{mode_object_label} DETECTED!")
                    object_detected.set()    # set the event
                
                
                
            # printing for debugging and logging purposes
            # tells us the time this iteration took, and the current iterations per second
            end_time = datetime.now()
            iteration_time = end_time - start_time
            iteration_time_seconds = str(iteration_time)[6:12]
            # our current version averages 13 frames processed per second (fpps), 
            # so printing every 40 frames equals around every 3 seconds
            if not object_detected.is_set():
                # we dont want it to fire while reacting to an object as the terminal would get too cluttered
                if (iter_count % 40) == 0:
                    print()
                    print(f"Iteration {iter_count}. Last iteration time: {iteration_time_seconds} seconds.")
                    print(f"Current frame rate: {round((1/float(iteration_time_seconds)),2)}fpps.")
                    print(f"Mode objected detected recently: {mode_object_label}")
                    print()
            
                    print("{label}, {size}, {score} of the last 3 objects:")
                    for box in object_data_stream[x-3:]:
                        print(f"{box[0]}, {box[1]}, {box[2]}")
                    print()

            iter_count += 1
    """End of while loop"""
    """End of C2"""
        
        
                    
