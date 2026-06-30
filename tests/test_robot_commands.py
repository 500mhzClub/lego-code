"""Syntax gate for the MicroPython sent to the LEGO hub.

The robot-side code lives as triple-quoted strings in control.py / movement.py /
misc.py and is shipped to the hub with ``pyb.exec(...)``. A typo in one of those
strings is invisible until it reaches the bot and blows up there. MicroPython is
a subset of Python, so we can ``compile()`` every snippet on the host and catch
syntax errors (and broken f-string interpolation) before any deploy.
"""

import pytest

from simulator import FakePyboard, collect_commands

COMMANDS = collect_commands()


def test_some_commands_were_discovered():
    # Guards against the discovery silently finding nothing (e.g. a refactor that
    # moves the builders elsewhere) and the suite passing vacuously.
    assert len(COMMANDS) >= 15


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_command_is_valid_python_syntax(name):
    source = COMMANDS[name]
    # mode="exec": these are full snippets (def/async def/top-level statements).
    compile(source, "<{}>".format(name), "exec")


def test_print_text_interpolation_is_safe():
    """A normal message round-trips into valid, well-formed MicroPython."""
    import misc

    pyb = FakePyboard()
    misc.print_text(pyb, "Hello")
    assert 'light_matrix.write("Hello")' in pyb.last
    compile(pyb.last, "<print_text>", "exec")


def test_no_command_is_accidentally_empty():
    for name, source in COMMANDS.items():
        assert source.strip(), "{} produced an empty command".format(name)
