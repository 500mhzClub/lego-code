"""Shared test fixtures and mocks.

The dataset/training utilities import heavy, device-only packages
(``tensorflow``, ``tflite_runtime`` and the TF Object Detection API). None of
those are needed to exercise the actual Python logic, so we install lightweight
fakes into ``sys.modules`` *before* the modules under test are imported. This
lets the whole suite run on any machine with just pandas/numpy/Pillow/pytest.
"""

import io
import sys
import types
from pathlib import Path

import pytest

# Make the project root importable (so `import create_csv` etc. works).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_fake_tensorflow():
    """A minimal stand-in for `tensorflow.compat.v1` good enough for create_tfrecord."""

    # --- tensorflow.python.framework.versions.VERSION ---
    _module("tensorflow")
    _module("tensorflow.python")
    _module("tensorflow.python.framework")
    versions = _module("tensorflow.python.framework.versions")
    versions.VERSION = "2.5.0"  # force the `import tensorflow.compat.v1` branch

    # --- the `tf` object itself (imported as tensorflow.compat.v1) ---
    tf = _module("tensorflow.compat.v1")
    _module("tensorflow.compat").v1 = tf
    sys.modules["tensorflow"].compat = sys.modules["tensorflow.compat"]

    # tf.train.Example / tf.train.Features -- capture the structure verbatim so
    # tests can assert on what was passed in.
    tf.train = types.SimpleNamespace(
        Example=lambda features: types.SimpleNamespace(features=features),
        Features=lambda feature: types.SimpleNamespace(feature=feature),
    )

    # tf.gfile.GFile -- the script only uses it to read a file in binary mode,
    # which the builtin open() does identically.
    tf.gfile = types.SimpleNamespace(GFile=open)

    # tf.python_io.TFRecordWriter -- collects whatever is written.
    class _Writer:
        def __init__(self, path):
            self.path = path
            self.records = []

        def write(self, record):
            self.records.append(record)

        def close(self):
            pass

    tf.python_io = types.SimpleNamespace(TFRecordWriter=_Writer)

    # tf.app.flags -- DEFINE_string is a no-op; FLAGS is a plain mutable holder
    # so tests can assign FLAGS.labelmap etc.
    flags_holder = types.SimpleNamespace()

    class _Flags:
        FLAGS = flags_holder

        @staticmethod
        def DEFINE_string(name, default, _help):
            setattr(flags_holder, name, default)

    tf.app = types.SimpleNamespace(flags=_Flags(), run=lambda *a, **k: None)
    return tf


def _install_fake_object_detection():
    """`object_detection.utils.dataset_util` feature helpers, as identity wrappers.

    Each helper normally returns a protobuf Feature; here it just returns the
    raw value so tests can read back exactly what the code computed.
    """
    _module("object_detection")
    _module("object_detection.utils")
    dataset_util = _module("object_detection.utils.dataset_util")
    for fn in (
        "int64_feature",
        "bytes_feature",
        "float_list_feature",
        "bytes_list_feature",
        "int64_list_feature",
    ):
        setattr(dataset_util, fn, (lambda value: value))
    return dataset_util


def _install_fake_tflite_runtime():
    runtime = _module("tflite_runtime")
    interpreter = _module("tflite_runtime.interpreter")
    interpreter.Interpreter = object
    interpreter.load_delegate = lambda *a, **k: None
    runtime.interpreter = interpreter


# Install the fakes at collection time, before any test imports the modules.
_install_fake_tensorflow()
_install_fake_object_detection()
_install_fake_tflite_runtime()


@pytest.fixture
def make_voc_xml(tmp_path):
    """Factory that writes a Pascal VOC annotation .xml file and returns its dir.

    Usage:
        folder = make_voc_xml("img1.jpg", 200, 100,
                              [("dime", 10, 20, 30, 40)])
    """

    def _factory(filename, width, height, objects, out_dir=None):
        out_dir = Path(out_dir or tmp_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        obj_xml = ""
        for name, xmin, ymin, xmax, ymax in objects:
            obj_xml += f"""
  <object>
    <name>{name}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""
        xml = f"""<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>{obj_xml}
</annotation>
"""
        xml_path = out_dir / (Path(filename).stem + ".xml")
        xml_path.write_text(xml)
        return out_dir

    return _factory


@pytest.fixture
def tiny_jpeg(tmp_path):
    """Write a real (small) JPEG and return (path, width, height)."""
    from PIL import Image

    width, height = 8, 4
    img = Image.new("RGB", (width, height), color=(123, 45, 67))
    path = tmp_path / "tiny.jpg"
    img.save(path, format="JPEG")
    return path, width, height
