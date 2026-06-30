"""Local, hardware-free simulator for Autonomous Lego.

The robot only ever touches three pieces of hardware:

    * the LEGO hub, via the ``Pyboard`` serial link (pyboard.py)
    * the camera, via ``VideoStream`` (videostream.py)
    * the Coral Edge TPU, via the tflite ``Interpreter``

Every one of those sits behind a small, well defined interface, so we can swap
in fakes and run the *real* ``driver()`` / ``detector()`` logic from streams.py
on a laptop -- no cable, no camera, no TPU. This is what lets you test a change
locally before sending code to the physical bot.

Run a dry run that prints the exact MicroPython that *would* be sent to the hub:

    python simulator.py

The fakes (``FakePyboard``, ``FakeVideoStream``, ``FakeInterpreter``) and the
helpers here are also imported by the test-suite in tests/.
"""

import sys
import types

import numpy as np


# --------------------------------------------------------------------------- #
# Device fakes
# --------------------------------------------------------------------------- #
class FakePyboard:
    """Drop-in for ``pyboard.Pyboard`` that records commands instead of sending.

    The real Pyboard serialises each command string over USB to the hub. Here we
    just keep them in ``self.commands`` so tests / the simulator can inspect the
    exact MicroPython that would have run on the bot.
    """

    def __init__(self, device="fake://", baudrate=115200, wait=0, verbose=False):
        self.device = device
        self.baudrate = baudrate
        self.commands = []          # every command string sent, in order
        self.in_raw_repl = False
        self.closed = False
        self.verbose = verbose

    # --- lifecycle (no-ops, just bookkeeping) ---
    def enter_raw_repl(self):
        self.in_raw_repl = True

    def exit_raw_repl(self):
        self.in_raw_repl = False

    def close(self):
        self.closed = True

    # --- execution (the bit that matters) ---
    def exec_(self, command, data_consumer=None):
        text = command.decode("utf8") if isinstance(command, (bytes, bytearray)) else command
        self.commands.append(text)
        if self.verbose:
            self._print_command(text)
        return b""

    # the real class aliases exec -> exec_ via setattr; do the same so callers
    # using ``pyb.exec(...)`` work unchanged.
    exec = exec_

    def eval(self, expression):
        return self.exec_("print({})".format(expression))

    def execfile(self, filename):
        with open(filename, "r") as f:
            return self.exec_(f.read())

    @property
    def last(self):
        return self.commands[-1] if self.commands else None

    def _print_command(self, text):
        body = "\n".join("    " + line for line in text.strip("\n").splitlines())
        print("\n--> SEND TO HUB " + "-" * 48)
        print(body)


class FakeVideoStream:
    """Drop-in for ``videostream.VideoStream`` returning blank frames."""

    def __init__(self, resolution=(1280, 720), framerate=30):
        self.width, self.height = resolution
        self.stopped = False

    def start(self):
        return self

    def read(self):
        # A real BGR frame; the detector rotates/resizes it. Content is irrelevant
        # because FakeInterpreter ignores the pixels and replays a script instead.
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def stop(self):
        self.stopped = True


class ScriptExhausted(Exception):
    """Raised by FakeInterpreter once its scripted frames run out.

    detector()'s inner ``while True`` has no natural exit, so this is how a dry
    run / test ends the loop cleanly after a fixed number of frames.
    """


class FakeInterpreter:
    """Drop-in for the tflite Interpreter that replays scripted detections.

    ``script`` is a list of frames; each frame is a list of detections, where a
    detection is ``(label, score, (ymin, xmin, ymax, xmax))`` with score in
    0..1 and box coords normalised to 0..1. An empty frame means "saw nothing".
    """

    # arbitrary but distinct tensor indices, mirroring a real model layout
    _BOXES, _CLASSES, _SCORES = 10, 11, 12

    def __init__(self, script, labels, raise_on_end=True, frame_delay=0.0):
        self.script = list(script)
        self.labels = labels
        self.raise_on_end = raise_on_end
        self.frame_delay = frame_delay  # pace frames so threads interleave realistically
        self._frame = 0
        self._current = []

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": [1, 300, 300, 3], "dtype": np.uint8}]

    def get_output_details(self):
        return [
            {"index": self._BOXES},
            {"index": self._CLASSES},
            {"index": self._SCORES},
        ]

    def set_tensor(self, index, data):
        pass

    def invoke(self):
        if self.frame_delay:
            import time
            time.sleep(self.frame_delay)
        if self._frame >= len(self.script):
            if self.raise_on_end:
                raise ScriptExhausted()
            self._current = []
            return
        self._current = self.script[self._frame]
        self._frame += 1

    def get_tensor(self, index):
        boxes, classes, scores = [], [], []
        for label, score, box in self._current:
            boxes.append(list(box))
            classes.append(float(self.labels.index(label)))
            scores.append(float(score))
        if index == self._BOXES:
            return np.array([boxes], dtype=float) if boxes else np.zeros((1, 0, 4))
        if index == self._CLASSES:
            return np.array([classes], dtype=float) if classes else np.zeros((1, 0))
        if index == self._SCORES:
            return np.array([scores], dtype=float) if scores else np.zeros((1, 0))
        raise KeyError(index)


# --------------------------------------------------------------------------- #
# Importing streams.py off-device
# --------------------------------------------------------------------------- #
def install_device_fakes():
    """Fake out cv2 / tflite_runtime so streams.py imports on a laptop.

    Only fakes a module if the real one is missing, so this is a no-op on the
    Raspberry Pi where the genuine packages are installed.
    """
    if "cv2" not in sys.modules:
        try:
            import cv2  # noqa: F401
        except ImportError:
            sys.modules["cv2"] = _make_fake_cv2()

    if "tflite_runtime" not in sys.modules:
        try:
            import tflite_runtime  # noqa: F401
        except ImportError:
            rt = types.ModuleType("tflite_runtime")
            interp = types.ModuleType("tflite_runtime.interpreter")
            interp.Interpreter = object
            interp.load_delegate = lambda *a, **k: None
            rt.interpreter = interp
            sys.modules["tflite_runtime"] = rt
            sys.modules["tflite_runtime.interpreter"] = interp


def _make_fake_cv2():
    cv2 = types.ModuleType("cv2")
    cv2.ROTATE_180 = 1
    cv2.COLOR_BGR2RGB = 4
    cv2.rotate = lambda img, code: img
    cv2.cvtColor = lambda img, code: img

    def resize(img, size):  # size is (width, height)
        w, h = size
        return np.zeros((h, w, img.shape[2] if img.ndim == 3 else 3), dtype=img.dtype)

    cv2.resize = resize
    return cv2


# --------------------------------------------------------------------------- #
# Command discovery (used by the syntax-gate test and the simulator)
# --------------------------------------------------------------------------- #
def collect_commands():
    """Call every hub-command builder with a FakePyboard and return its output.

    Returns ``{ "module.function": command_string, ... }`` covering every
    MicroPython snippet the robot can send. Functions that take extra arguments
    (e.g. ``misc.print_text(pyb, message)``) are given a placeholder value.
    """
    import inspect
    import control
    import misc
    import movement

    placeholders = {"message": "HELLO"}
    out = {}
    for module in (control, movement, misc):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != module.__name__:
                continue  # skip imported helpers, only this module's own funcs
            params = list(inspect.signature(fn).parameters)
            pyb = FakePyboard()
            kwargs = {p: placeholders.get(p, "TEST") for p in params[1:]}
            fn(pyb, **kwargs)
            for i, cmd in enumerate(pyb.commands):
                key = "{}.{}".format(module.__name__, name)
                if len(pyb.commands) > 1:
                    key += "[{}]".format(i)
                out[key] = cmd
    return out


# --------------------------------------------------------------------------- #
# Dry-run simulation
# --------------------------------------------------------------------------- #
def default_labels():
    # mirrors the on-device labelmap.txt shape: real classes plus UNKNOWN padding
    return [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    ] + ["UNKNOWN"] * 5


def _big_box():
    # ~0.6 x 0.6 of frame -> well over detector's min_size_threshold
    return (0.2, 0.2, 0.8, 0.8)


def demo_script(labels):
    """A scripted drive: cruise, see a person, then hit a stop sign."""
    quiet = [[] for _ in range(20)]
    person = [[("person", 0.9, _big_box())] for _ in range(15)]
    stop = [[("stop sign", 0.95, _big_box())] for _ in range(15)]
    return quiet + person + quiet + stop + quiet


def run_simulation(script=None, labels=None, verbose=True):
    """Run the REAL driver()/detector() against fakes and report hub commands."""
    import queue
    import threading
    import time

    install_device_fakes()
    import streams
    from streams import driver, detector  # imported after fakes are in place

    # The real reactions sleep for up to 6 real seconds; cap them so the dry run
    # stays quick while still letting the driver act each reaction out in order.
    real_sleep = time.sleep
    streams.time.sleep = lambda s=0: real_sleep(min(s, 0.05))

    labels = labels or default_labels()
    script = script if script is not None else demo_script(labels)

    pyb = FakePyboard(verbose=verbose)
    videostream = FakeVideoStream()
    # ~50 fps of scripted frames -> driver has time to react between detections
    interpreter = FakeInterpreter(script, labels, raise_on_end=True, frame_delay=0.02)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    width = height = 300
    imW, imH = videostream.width, videostream.height

    object_detected = threading.Event()
    shutdown_event = threading.Event()
    q = queue.Queue()

    def safe_driver():
        try:
            driver(object_detected, shutdown_event, q, pyb)
        except KeyboardInterrupt:
            pass  # driver signals its own completion this way

    def safe_detector():
        try:
            detector(object_detected, shutdown_event, pyb, videostream, width,
                     height, interpreter, input_details, output_details,
                     imW, imH, labels, q)
        except ScriptExhausted:
            pass  # ran out of scripted frames -> end of the dry run

    print("=== Autonomous Lego dry run ({} frames scripted) ===".format(len(script)))
    dt = threading.Thread(target=safe_driver)
    et = threading.Thread(target=safe_detector)
    dt.start()
    et.start()

    et.join(timeout=30)          # detector finishes when the script is exhausted
    shutdown_event.set()         # tell the driver to wind down
    dt.join(timeout=5)

    print("\n=== Dry run complete: {} command(s) sent to the (fake) hub ===".format(len(pyb.commands)))
    return pyb.commands


if __name__ == "__main__":
    run_simulation()
