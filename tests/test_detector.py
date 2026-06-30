"""Behaviour tests for streams.detector() -- the perception/voting logic.

detector() aggregates the last N frames, finds the mode object, and only fires
when a real, close, confident object has persisted long enough. We drive the
real detector() with a FakeInterpreter that replays scripted detections, so we
can check the voting/threshold logic without a camera or a TPU.
"""

import queue
import threading

import streams
from simulator import (
    FakeInterpreter,
    FakePyboard,
    FakeVideoStream,
    ScriptExhausted,
    default_labels,
)

LABELS = default_labels()


def _big_box():
    # ~0.6 x 0.6 of a 1280x720 frame -> ~330k px, far above min_size_threshold.
    return (0.2, 0.2, 0.8, 0.8)


def _run_detector(script):
    """Run detector() over a finite script; return (object_detected, queued)."""
    pyb = FakePyboard()
    videostream = FakeVideoStream()
    interpreter = FakeInterpreter(script, LABELS, raise_on_end=True)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    object_detected = threading.Event()
    shutdown_event = threading.Event()
    q = queue.Queue()

    try:
        streams.detector(object_detected, shutdown_event, pyb, videostream,
                         300, 300, interpreter, input_details, output_details,
                         videostream.width, videostream.height, LABELS, q)
    except ScriptExhausted:
        pass

    queued = list(q.queue)
    return object_detected, queued


def test_persistent_close_object_fires():
    script = [[("stop sign", 0.95, _big_box())] for _ in range(20)]
    object_detected, queued = _run_detector(script)
    assert object_detected.is_set()
    assert "stop sign" in queued


def test_empty_frames_never_fire():
    script = [[] for _ in range(20)]
    object_detected, queued = _run_detector(script)
    assert not object_detected.is_set()
    assert queued == []


def test_low_confidence_is_ignored():
    # Below min_conf_threshold (0.3): treated as "None", never reacts.
    script = [[("stop sign", 0.1, _big_box())] for _ in range(20)]
    object_detected, queued = _run_detector(script)
    assert not object_detected.is_set()


def test_far_away_object_is_ignored():
    # Confident but tiny box -> under min_size_threshold, so it stays cruising.
    tiny = (0.50, 0.50, 0.52, 0.52)
    script = [[("stop sign", 0.95, tiny)] for _ in range(20)]
    object_detected, queued = _run_detector(script)
    assert not object_detected.is_set()


def test_unimplemented_class_queues_unknown_object_event():
    # "car" is detected by the model but isn't in implemented_classes, so the
    # driver can confirm it with the distance sensor before reacting.
    script = [[("car", 0.95, _big_box())] for _ in range(20)]
    object_detected, queued = _run_detector(script)
    assert object_detected.is_set()
    assert queued == [{"class": streams.UNKNOWN_OBJECT, "label": "car"}]


def test_only_one_event_until_cleared():
    # Even with a long run of the same object, exactly one event is raised until
    # the flag is cleared (driver()'s job), so the queue holds a single label.
    script = [[("stop sign", 0.95, _big_box())] for _ in range(40)]
    object_detected, queued = _run_detector(script)
    assert queued == ["stop sign"]
