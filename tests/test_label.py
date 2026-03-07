"""Tests for the labeling engine."""

import csv

from labelman.label import (
    apply_labels,
    labels_to_caption,
    write_csv,
    write_report,
    write_sidecar,
)
from labelman.schema import parse


def _parse(yaml_str):
    return parse(yaml_str)


TAXONOMY = """\
defaults:
  threshold: 0.3
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: setting
    mode: zero-or-one
    threshold: 0.45
    terms:
      - term: indoor
      - term: outdoor
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
      - term: tense
        threshold: 0.25
"""


# --- Core labeling ---

def test_label_exactly_one_above_threshold():
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.8, "group": 0.2}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["count"] == ["single"]


def test_label_exactly_one_below_threshold():
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.1, "group": 0.05}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["count"] == ["single"]


def test_label_zero_or_one_above_threshold():
    tl = _parse(TAXONOMY)
    scores = {"setting": {"indoor": 0.6, "outdoor": 0.3}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["setting"] == ["indoor"]


def test_label_zero_or_one_below_threshold():
    tl = _parse(TAXONOMY)
    scores = {"setting": {"indoor": 0.4, "outdoor": 0.3}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["setting"] == []


def test_label_zero_or_more_multiple():
    tl = _parse(TAXONOMY)
    scores = {"mood": {"calm": 0.5, "energetic": 0.4, "tense": 0.27}}
    result = apply_labels(tl, "img.jpg", scores)
    assert set(result.labels["mood"]) == {"calm", "energetic", "tense"}


def test_label_zero_or_more_none():
    tl = _parse(TAXONOMY)
    scores = {"mood": {"calm": 0.1, "energetic": 0.05, "tense": 0.1}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["mood"] == []


def test_label_zero_or_more_all():
    tl = _parse(TAXONOMY)
    scores = {"mood": {"calm": 0.9, "energetic": 0.8, "tense": 0.5}}
    result = apply_labels(tl, "img.jpg", scores)
    assert set(result.labels["mood"]) == {"calm", "energetic", "tense"}


def test_label_threshold_inheritance():
    tl = _parse(TAXONOMY)
    scores = {"mood": {"calm": 0.1, "energetic": 0.28, "tense": 0.26}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["mood"] == ["tense"]


def test_label_deterministic():
    tl = _parse(TAXONOMY)
    scores = {
        "count": {"single": 0.7, "group": 0.3},
        "setting": {"indoor": 0.5, "outdoor": 0.4},
        "mood": {"calm": 0.6, "energetic": 0.1, "tense": 0.3},
    }
    r1 = apply_labels(tl, "img.jpg", scores)
    r2 = apply_labels(tl, "img.jpg", scores)
    assert r1.labels == r2.labels


def test_label_multi_category():
    tl = _parse(TAXONOMY)
    scores = {
        "count": {"single": 0.9, "group": 0.1},
        "setting": {"indoor": 0.5, "outdoor": 0.2},
        "mood": {"calm": 0.4, "energetic": 0.1, "tense": 0.3},
    }
    result = apply_labels(tl, "img.jpg", scores)
    assert result.labels["count"] == ["single"]
    assert result.labels["setting"] == ["indoor"]
    assert "calm" in result.labels["mood"]
    assert "tense" in result.labels["mood"]


def test_label_multi_image():
    tl = _parse(TAXONOMY)
    s1 = {"count": {"single": 0.9, "group": 0.1}}
    s2 = {"count": {"single": 0.1, "group": 0.9}}
    r1 = apply_labels(tl, "img1.jpg", s1)
    r2 = apply_labels(tl, "img2.jpg", s2)
    assert r1.labels["count"] == ["single"]
    assert r2.labels["count"] == ["group"]
    assert r1.image == "img1.jpg"
    assert r2.image == "img2.jpg"


def test_label_missing_scores_default_zero():
    tl = _parse(TAXONOMY)
    scores = {}
    result = apply_labels(tl, "img.jpg", scores)
    assert len(result.labels["count"]) == 1
    assert result.labels["setting"] == []
    assert result.labels["mood"] == []


def test_label_preserves_all_scores():
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.8, "group": 0.2}, "mood": {"calm": 0.1}}
    result = apply_labels(tl, "img.jpg", scores)
    assert result.all_scores == scores


# --- Caption generation ---

def test_labels_to_caption():
    tl = _parse(TAXONOMY)
    scores = {
        "count": {"single": 0.9, "group": 0.1},
        "setting": {"indoor": 0.5, "outdoor": 0.2},
        "mood": {"calm": 0.4, "energetic": 0.1, "tense": 0.3},
    }
    result = apply_labels(tl, "img.jpg", scores)
    caption = labels_to_caption(result)
    assert "single" in caption
    assert "indoor" in caption
    assert ", " in caption


def test_labels_to_caption_empty_categories():
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    result = apply_labels(tl, "img.jpg", scores)
    caption = labels_to_caption(result)
    assert "single" in caption


# --- Sidecar files ---

def test_write_sidecar(tmp_path):
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    img = tmp_path / "photo.jpg"
    img.write_text("")  # dummy image
    result = apply_labels(tl, str(img), scores)
    sidecar = write_sidecar(result)
    assert sidecar.suffix == ".txt"
    assert sidecar.stem == "photo"
    content = sidecar.read_text()
    assert "single" in content


def test_write_sidecar_output_dir(tmp_path):
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    result = apply_labels(tl, "/data/images/photo.jpg", scores)
    out_dir = tmp_path / "labels"
    sidecar = write_sidecar(result, output_dir=out_dir)
    assert sidecar.parent == out_dir
    assert sidecar.name == "photo.txt"
    assert sidecar.read_text()


# --- CSV output ---

def test_write_csv(tmp_path):
    tl = _parse(TAXONOMY)
    scores1 = {
        "count": {"single": 0.9, "group": 0.1},
        "setting": {"indoor": 0.5, "outdoor": 0.2},
        "mood": {"calm": 0.4, "energetic": 0.1, "tense": 0.3},
    }
    scores2 = {
        "count": {"single": 0.2, "group": 0.8},
        "mood": {"calm": 0.1, "energetic": 0.9, "tense": 0.05},
    }
    r1 = apply_labels(tl, "img1.jpg", scores1)
    r2 = apply_labels(tl, "img2.jpg", scores2)

    csv_path = tmp_path / "labels.csv"
    write_csv(tl, [r1, r2], csv_path)

    with csv_path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    assert header[0] == "image"
    assert header[-1] == "caption"
    assert "count/single_score" in header
    assert "count/single_assigned" in header
    assert len(rows) == 3  # header + 2 images


def test_write_csv_includes_suppressed_scores(tmp_path):
    tl = _parse(TAXONOMY)
    scores = {
        "count": {"single": 0.9, "group": 0.1},
        "mood": {"calm": 0.1, "energetic": 0.05, "tense": 0.1},
    }
    result = apply_labels(tl, "img.jpg", scores)

    csv_path = tmp_path / "labels.csv"
    write_csv(tl, [result], csv_path)

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # group was not assigned but its score should be present
    assert float(row["count/group_score"]) == 0.1
    assert row["count/group_assigned"] == "0"
    # calm was below threshold, suppressed
    assert float(row["mood/calm_score"]) == 0.1
    assert row["mood/calm_assigned"] == "0"


# --- HTML report ---

def test_write_report(tmp_path):
    tl = _parse(TAXONOMY)
    scores1 = {
        "count": {"single": 0.9, "group": 0.1},
        "mood": {"calm": 0.4, "energetic": 0.1, "tense": 0.3},
    }
    scores2 = {
        "count": {"single": 0.2, "group": 0.8},
        "mood": {"calm": 0.5, "energetic": 0.9, "tense": 0.05},
    }
    r1 = apply_labels(tl, "img1.jpg", scores1)
    r2 = apply_labels(tl, "img2.jpg", scores2)

    report_path = tmp_path / "report.html"
    write_report(tl, [r1, r2], report_path)

    content = report_path.read_text()
    assert "<!DOCTYPE html>" in content
    assert "labelman Report" in content
    # Summary section
    assert "Images: 2" in content
    assert "Categories: 3" in content
    # Category stats
    assert "count" in content
    assert "single" in content
    # Per-image detail
    assert "img1.jpg" in content
    assert "img2.jpg" in content


def test_write_report_has_category_stats(tmp_path):
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    result = apply_labels(tl, "img.jpg", scores)

    report_path = tmp_path / "report.html"
    write_report(tl, [result], report_path)

    content = report_path.read_text()
    assert "100.0%" in content  # single assigned to 100% of images
    assert "0.0%" in content    # group assigned to 0%
