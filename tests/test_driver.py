"""Behaviour tests for streams.driver() -- the robot's decision loop.

driver() reads a detected class off the queue and reacts by sending movement
commands to the hub. We run the *real* driver() against a FakePyboard so we can
assert "when the bot sees X, it sends Y" without any hardware -- and without the
multi-second time.sleep() calls the real reactions use (we patch them out).
"""

import queue
import threading
import time

import pytest

import movement
import streams


def _run_driver_for_class(class_detected, monkeypatch, timeout=3.0):
    """Drive a single detection through driver() and return commands sent.

    Returns the list of MicroPython command strings the FakePyboard captured.
    """
    from simulator import FakePyboard

    # Reactions sleep for several real seconds (e.g. 6s for "person"); no-op them
    # so the test is fast and deterministic.
    monkeypatch.setattr(streams.time, "sleep", lambda *a, **k: None)

    pyb = FakePyboard()
    object_detected = threading.Event()
    shutdown_event = threading.Event()
    q = queue.Queue()

    q.put(class_detected)
    object_detected.set()

    def target():
        try:
            streams.driver(object_detected, shutdown_event, q, pyb)
        except KeyboardInterrupt:
            pass  # driver raises this to signal its own shutdown

    t = threading.Thread(target=target)
    t.start()

    # Wait until the detection has been consumed (flag cleared) or the driver
    # shut itself down (the "stop sign" path).
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not object_detected.is_set() or shutdown_event.is_set():
            break
        time.sleep(0.01)

    shutdown_event.set()
    t.join(timeout=timeout)
    assert not t.is_alive(), "driver thread did not terminate"
    return pyb.commands


def _joined(commands):
    return "\n".join(commands)


def test_driver_starts_by_driving(monkeypatch):
    cmds = _run_driver_for_class("person", monkeypatch)
    # First thing the driver ever does is start moving.
    assert "mp.move(mp.PAIR_1" in cmds[0]


def test_stop_sign_halts_with_hazards_and_shuts_down(monkeypatch):
    from simulator import FakePyboard

    monkeypatch.setattr(streams.time, "sleep", lambda *a, **k: None)
    pyb = FakePyboard()
    object_detected = threading.Event()
    shutdown_event = threading.Event()
    q = queue.Queue()
    q.put("stop sign")
    object_detected.set()

    def target():
        try:
            streams.driver(object_detected, shutdown_event, q, pyb)
        except KeyboardInterrupt:
            pass

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=3.0)

    assert not t.is_alive()
    # A stop sign irreversibly halts the robot...
    assert shutdown_event.is_set()
    # ...via the hazards stop sequence (distinct light_matrix.show_image(1)).
    assert any("light_matrix.show_image(1)" in c for c in pyb.commands)


def test_close_distance_sensor_obstacle_halts_with_hazards(monkeypatch):
    from simulator import FakePyboard

    monkeypatch.setattr(streams.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(movement, "read_distance_mm", lambda pyb: 100)

    pyb = FakePyboard()
    object_detected = threading.Event()
    shutdown_event = threading.Event()
    q = queue.Queue()

    def target():
        try:
            streams.driver(object_detected, shutdown_event, q, pyb)
        except KeyboardInterrupt:
            pass

    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=3.0)

    assert not t.is_alive()
    assert shutdown_event.is_set()
    assert any("light_matrix.show_image(1)" in c for c in pyb.commands)


def test_traffic_light_stops_then_turns(monkeypatch):
    cmds = _joined(_run_driver_for_class("traffic light", monkeypatch))
    # turn_right_gyro is identifiable by its gyro threshold loop.
    assert "motion_sensor.tilt_angles()[0] > -900" in cmds


def test_person_triggers_headlight_flash_and_slow(monkeypatch):
    cmds = _joined(_run_driver_for_class("person", monkeypatch))
    assert "full_beam" in cmds                 # headlight_flash
    assert "velocity = 150" in cmds            # drive_slow


def test_bicycle_triggers_overtake(monkeypatch):
    cmds = _joined(_run_driver_for_class("bicycle", monkeypatch))
    # overtake_right is the only reaction that pairs and reverses tank steering.
    assert "move_tank" in cmds


def test_cup_triggers_fast_drive(monkeypatch):
    cmds = _joined(_run_driver_for_class("cup", monkeypatch))
    assert "velocity = 700" in cmds


def test_unknown_class_is_ignored_gracefully(monkeypatch, capsys):
    cmds = _run_driver_for_class("giraffe", monkeypatch)
    out = capsys.readouterr().out
    assert "UNKNOWN OBJECTED DETECTED: giraffe" in out
    # It still recovers and resumes driving afterwards.
    assert any("mp.move(mp.PAIR_1" in c for c in cmds)
