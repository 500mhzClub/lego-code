#!/bin/bash
# Run the unit test suite.
#
# First-time setup:
#   python3 -m venv .venv
#   .venv/bin/pip install -r requirements-dev.txt
#
# Then just: ./run_tests.sh   (extra args are passed through to pytest)

set -e
cd "$(dirname "$0")"

# This machine has system-wide ROS pytest plugins (launch_testing) that fail to
# import in a clean venv. We don't use them, so disable third-party autoload.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

exec "$PY" -m pytest "$@"
