"""Tests for create_csv.xml_to_csv (pure pandas, no mocking needed)."""

import create_csv


def test_single_object(make_voc_xml):
    folder = make_voc_xml("img1.jpg", 200, 100, [("dime", 10, 20, 30, 40)])

    df = create_csv.xml_to_csv(str(folder))

    assert list(df.columns) == [
        "filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["filename"] == "img1.jpg"
    assert (row["width"], row["height"]) == (200, 100)
    assert row["class"] == "dime"
    assert (row["xmin"], row["ymin"], row["xmax"], row["ymax"]) == (10, 20, 30, 40)


def test_multiple_objects_in_one_file(make_voc_xml):
    folder = make_voc_xml(
        "img2.jpg", 640, 480,
        [("penny", 1, 2, 3, 4), ("nickel", 5, 6, 7, 8)],
    )

    df = create_csv.xml_to_csv(str(folder))

    assert len(df) == 2
    assert set(df["class"]) == {"penny", "nickel"}


def test_multiple_files_are_aggregated(make_voc_xml, tmp_path):
    make_voc_xml("a.jpg", 100, 100, [("dime", 1, 1, 2, 2)], out_dir=tmp_path)
    make_voc_xml("b.jpg", 100, 100, [("dime", 3, 3, 4, 4)], out_dir=tmp_path)

    df = create_csv.xml_to_csv(str(tmp_path))

    assert len(df) == 2
    assert set(df["filename"]) == {"a.jpg", "b.jpg"}


def test_empty_folder_returns_empty_frame(tmp_path):
    df = create_csv.xml_to_csv(str(tmp_path))

    assert len(df) == 0
    # Columns are still defined so a downstream to_csv() produces a valid header.
    assert "filename" in df.columns


def test_coordinates_are_integers(make_voc_xml):
    folder = make_voc_xml("img.jpg", 50, 50, [("dime", 11, 12, 13, 14)])

    df = create_csv.xml_to_csv(str(folder))

    for col in ("width", "height", "xmin", "ymin", "xmax", "ymax"):
        assert isinstance(df.iloc[0][col], (int,)) or str(df[col].dtype).startswith("int")
