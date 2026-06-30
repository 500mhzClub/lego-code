## Utility Scripts for TensorFlow Lite Object Detection
These scripts are used in the [TFLite Training Colab](https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb) to help with various steps of training a custom model. They can also be used as standalone tools.

### Calculate model mAP - (calculate_map_cartucho.py)
Calculate your TFLite detection model's mAP score! I'll share instructions on how to use this outside the Colab notebook later. 

<img src="../doc/calculate-mAP-demo1.gif">

This tool uses the main.py script from [Cartucho's excellent repository](https://github.com/Cartucho/mAP), which takes in ground truth data and detection results to calculate average precision at a certain IoU threshold. The calculate_map_cartucho.py script performs the mAP calculation at multiple IoU thresholds to determine the COCO metric for average mAP @ 0.5:0.95.

### Split images into train, test, and validation sets - (train_val_test.py)
This script takes a folder full of images and randomly splits them between train, test, and validation folders. It does an 80%/10%/10% split by default, but this can be modified by changing the `train_percent`, `test_percent`, and `val_percent` variables in the code.

### Create CSV annotation file - (create_csv.py)
This script creates a single CSV data file from a set of Pascal VOC annotation files.

Original credit for the script goes to [datitran](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py).

### Create TFRecord file - (create_tfrecord.py)
This script creates TFRecord files from a CSV annotation data file and a folder of images. TFRecords are the [data format required](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md) by the TensorFlow Object Detection API for training.

Original credit for the script goes to [datitran](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py).
# lego-code

## Testing changes locally before deploying to the bot

The robot streams MicroPython source to the LEGO hub over serial, and runs
object detection on the Pi. None of that logic actually *needs* the hub, camera
or Coral TPU to be validated -- each sits behind a small interface that we fake
in [simulator.py](simulator.py). This lets you catch most mistakes on a laptop
before sending code to the physical bot.

First-time setup:

    python3 -m venv .venv
    .venv/bin/pip install -r requirements-dev.txt

**Run the test suite** (syntax-checks every hub command + exercises the
driver/detector decision logic):

    ./run_tests.sh

**Dry-run the whole robot** against fakes -- prints the exact MicroPython that
*would* be sent to the hub for a scripted drive (cruise, see a person, hit a
stop sign):

    .venv/bin/python simulator.py

What this catches without hardware:

* syntax errors / typos in the MicroPython strings in `control/movement/misc.py`
  (`tests/test_robot_commands.py` compiles every one)
* wrong reactions in `driver()` -- "when the bot sees X, send Y"
  (`tests/test_driver.py`)
* the frame-voting / distance / confidence thresholds in `detector()`
  (`tests/test_detector.py`)

What still needs the real bot: physical motor ports, velocities and gyro angles
(the tests assert the *command text* is what you intended, not how it moves).
