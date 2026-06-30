"""Tests for create_tfrecord.

tensorflow and the object_detection API are mocked in conftest.py, so this
exercises the real grouping logic and bbox-normalization math without needing
those packages installed.
"""

import pandas as pd
import pytest

import create_tfrecord


def test_split_groups_rows_by_filename():
    df = pd.DataFrame(
        {
            "filename": ["a.jpg", "a.jpg", "b.jpg"],
            "class": ["dime", "penny", "dime"],
            "width": [10, 10, 20],
            "height": [10, 10, 20],
            "xmin": [1, 2, 3],
            "ymin": [1, 2, 3],
            "xmax": [4, 5, 6],
            "ymax": [4, 5, 6],
        }
    )

    groups = create_tfrecord.split(df, "filename")

    by_name = {g.filename: g for g in groups}
    assert set(by_name) == {"a.jpg", "b.jpg"}
    assert len(by_name["a.jpg"].object) == 2
    assert len(by_name["b.jpg"].object) == 1


def test_create_tf_example_normalizes_bboxes(tmp_path, tiny_jpeg):
    img_path, width, height = tiny_jpeg  # 8 x 4 jpeg
    image_dir = img_path.parent

    # labelmap: dime is the 2nd label -> 1-based class id == 2
    labelmap = tmp_path / "labelmap.txt"
    labelmap.write_text("penny\ndime\n")
    create_tfrecord.FLAGS.labelmap = str(labelmap)

    df = pd.DataFrame(
        {
            "filename": [img_path.name],
            "class": ["dime"],
            "xmin": [2], "ymin": [1], "xmax": [4], "ymax": [3],
        }
    )
    group = create_tfrecord.split(df, "filename")[0]

    example = create_tfrecord.create_tf_example(group, str(image_dir))
    feat = example.features.feature  # identity mocks -> raw values

    assert feat["image/height"] == height
    assert feat["image/width"] == width
    assert feat["image/filename"] == img_path.name.encode("utf8")
    # Coordinates are normalized to [0, 1] against the image dimensions.
    assert feat["image/object/bbox/xmin"] == [2 / width]
    assert feat["image/object/bbox/xmax"] == [4 / width]
    assert feat["image/object/bbox/ymin"] == [1 / height]
    assert feat["image/object/bbox/ymax"] == [3 / height]
    assert feat["image/object/class/text"] == [b"dime"]
    assert feat["image/object/class/label"] == [2]


def test_create_tf_example_handles_multiple_objects(tmp_path, tiny_jpeg):
    img_path, width, height = tiny_jpeg
    labelmap = tmp_path / "labelmap.txt"
    labelmap.write_text("penny\ndime\n")
    create_tfrecord.FLAGS.labelmap = str(labelmap)

    df = pd.DataFrame(
        {
            "filename": [img_path.name, img_path.name],
            "class": ["penny", "dime"],
            "xmin": [0, 2], "ymin": [0, 1], "xmax": [4, 8], "ymax": [2, 4],
        }
    )
    group = create_tfrecord.split(df, "filename")[0]

    example = create_tfrecord.create_tf_example(group, str(image_dir := img_path.parent))
    feat = example.features.feature

    assert feat["image/object/class/text"] == [b"penny", b"dime"]
    assert feat["image/object/class/label"] == [1, 2]
    assert feat["image/object/bbox/xmax"] == [4 / width, 8 / width]
