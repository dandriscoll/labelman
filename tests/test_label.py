"""Tests for the labeling engine."""

import csv

from labelman.label import (
    apply_labels,
    assemble_final_labels,
    labels_to_caption,
    load_manual_sidecar,
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


# --- Manual label sidecars ---

def test_load_manual_sidecar_missing(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    labels = load_manual_sidecar(str(img))
    assert labels == []


def test_load_manual_sidecar_valid(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    sidecar = tmp_path / "photo.labels.txt"
    sidecar.write_text("tail number n123ab\nred stripe livery\n")
    labels = load_manual_sidecar(str(img))
    assert labels == ["tail number n123ab", "red stripe livery"]


def test_load_manual_sidecar_whitespace_and_blanks(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    sidecar = tmp_path / "photo.labels.txt"
    sidecar.write_text("  tail number n123ab  \n\n\n  red stripe  \n  \n")
    labels = load_manual_sidecar(str(img))
    assert labels == ["tail number n123ab", "red stripe"]


def test_load_manual_sidecar_empty_file(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    sidecar = tmp_path / "photo.labels.txt"
    sidecar.write_text("")
    labels = load_manual_sidecar(str(img))
    assert labels == []


# --- Final label assembly ---

TAXONOMY_WITH_GLOBALS = """\
defaults:
  threshold: 0.3
global_terms:
  - aircraft
  - mooney m20
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
"""


def test_assemble_global_terms_prepended():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.1}}
    result = assemble_final_labels(tl, "img.jpg", scores)
    assert result.final_labels[0] == "aircraft"
    assert result.final_labels[1] == "mooney m20"
    assert "single" in result.final_labels
    assert "calm" in result.final_labels


def test_assemble_manual_labels_merged():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["tail number n123ab", "red stripe livery"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "tail number n123ab" in result.final_labels
    assert "red stripe livery" in result.final_labels


def test_assemble_order_global_manual_detected():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["custom tag"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    # Order: global terms, manual, detected
    idx_aircraft = result.final_labels.index("aircraft")
    idx_mooney = result.final_labels.index("mooney m20")
    idx_custom = result.final_labels.index("custom tag")
    idx_single = result.final_labels.index("single")
    assert idx_aircraft < idx_mooney < idx_custom < idx_single


def test_assemble_deduplication():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.1}}
    # manual label duplicates a global term and a detected label
    manual = ["aircraft", "single", "custom"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    # Each appears exactly once
    assert result.final_labels.count("aircraft") == 1
    assert result.final_labels.count("single") == 1
    assert result.final_labels.count("custom") == 1


def test_assemble_manual_outside_taxonomy():
    """Manual labels not in the taxonomy are still included."""
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["not in taxonomy", "another custom"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "not in taxonomy" in result.final_labels
    assert "another custom" in result.final_labels


def test_assemble_detected_still_obey_thresholds():
    """Detected labels still respect category rules even with globals/manual."""
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.1, "energetic": 0.05}}
    result = assemble_final_labels(tl, "img.jpg", scores)
    # mood terms below threshold should not appear
    assert "calm" not in result.final_labels
    assert "energetic" not in result.final_labels
    # exactly-one still forces assignment
    assert "single" in result.final_labels


def test_assemble_no_manual_no_global():
    """Without globals or manuals, final_labels equals detected labels."""
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.1}}
    result = assemble_final_labels(tl, "img.jpg", scores)
    assert result.final_labels == ["single", "calm"]


def test_assemble_deterministic():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.4}}
    manual = ["custom tag"]
    r1 = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    r2 = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert r1.final_labels == r2.final_labels


def test_assemble_caption_includes_all_sources():
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["custom tag"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    caption = labels_to_caption(result)
    assert "aircraft" in caption
    assert "mooney m20" in caption
    assert "custom tag" in caption
    assert "single" in caption
    assert ", " in caption
