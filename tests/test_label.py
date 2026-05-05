"""Tests for the labeling engine."""

import csv

from labelman.label import (
    MB_FORMAT,
    TXT_FORMAT,
    apply_labels,
    assemble_final_labels,
    labels_to_caption,
    load_detected_sidecar,
    load_manual_sidecar,
    merge_sidecars,
    write_csv,
    write_final_sidecar,
    write_manual_sidecar,
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
    assert sidecar.name == "photo.detected.txt"
    content = sidecar.read_text()
    assert "single" in content


def test_write_sidecar_output_dir(tmp_path):
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    result = apply_labels(tl, "/data/images/photo.jpg", scores)
    out_dir = tmp_path / "labels"
    sidecar = write_sidecar(result, output_dir=out_dir)
    assert sidecar.parent == out_dir
    assert sidecar.name == "photo.detected.txt"
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
    sidecar.write_text("tail number n123ab, red stripe livery")
    labels = load_manual_sidecar(str(img))
    assert labels == ["tail number n123ab", "red stripe livery"]


def test_load_manual_sidecar_whitespace_and_blanks(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    sidecar = tmp_path / "photo.labels.txt"
    sidecar.write_text("  tail number n123ab  ,  ,  red stripe  ,  ")
    labels = load_manual_sidecar(str(img))
    assert labels == ["tail number n123ab", "red stripe"]


def test_load_manual_sidecar_empty_file(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_text("")
    sidecar = tmp_path / "photo.labels.txt"
    sidecar.write_text("")
    labels = load_manual_sidecar(str(img))
    assert labels == []


# --- Merge sidecars ---

def test_merge_sidecars_detected_only(tmp_path):
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.detected.txt").write_text("single, calm")
    merged = merge_sidecars(tl, str(img))
    # global terms + detected
    assert merged == ["aircraft", "mooney m20", "single", "calm"]


def test_merge_sidecars_manual_and_detected(tmp_path):
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.detected.txt").write_text("single, calm")
    (tmp_path / "photo.labels.txt").write_text("custom tag")
    merged = merge_sidecars(tl, str(img))
    assert merged == ["aircraft", "mooney m20", "custom tag", "single", "calm"]


def test_merge_sidecars_suppression(tmp_path):
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.detected.txt").write_text("single, calm")
    (tmp_path / "photo.labels.txt").write_text("-calm")
    merged = merge_sidecars(tl, str(img))
    assert "calm" not in merged
    assert "single" in merged


def test_merge_sidecars_no_sidecars(tmp_path):
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    merged = merge_sidecars(tl, str(img))
    # Only global terms
    assert merged == ["aircraft", "mooney m20"]


def test_write_final_sidecar(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    sidecar = write_final_sidecar(str(img), ["aircraft", "single", "calm"])
    assert sidecar.name == "photo.txt"
    assert sidecar.read_text() == "aircraft, single, calm"


def test_write_final_sidecar_output_dir(tmp_path):
    out_dir = tmp_path / "output"
    sidecar = write_final_sidecar("/data/images/photo.jpg", ["aircraft"], output_dir=out_dir)
    assert sidecar == out_dir / "photo.txt"
    assert sidecar.read_text() == "aircraft"


# --- Markback (.mb) sidecar format ---

def test_mb_format_encode_decode_round_trip():
    labels = ["outdoor", "single", "calm"]
    encoded = MB_FORMAT.encode(labels)
    # Markback sidecar form: leading "<<<" delimits feedback in V1 paired mode.
    assert "<<<" in encoded
    assert "outdoor; single; calm" in encoded
    assert MB_FORMAT.decode(encoded) == labels


def test_mb_format_encode_empty_returns_empty_string():
    assert MB_FORMAT.encode([]) == ""
    assert MB_FORMAT.decode("") == []


def test_mb_format_decode_strips_whitespace():
    text = "<<< outdoor ;  single  ; calm \n"
    assert MB_FORMAT.decode(text) == ["outdoor", "single", "calm"]


def test_txt_format_encode_decode_round_trip():
    labels = ["outdoor", "single", "calm"]
    assert TXT_FORMAT.encode(labels) == "outdoor, single, calm"
    assert TXT_FORMAT.decode(TXT_FORMAT.encode(labels)) == labels


def test_load_manual_sidecar_mb(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.labels.mb").write_text("<<< outdoor; single; calm\n")
    assert load_manual_sidecar(str(img), fmt=MB_FORMAT) == ["outdoor", "single", "calm"]


def test_load_manual_sidecar_mb_does_not_read_txt(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.labels.txt").write_text("outdoor, single")
    assert load_manual_sidecar(str(img), fmt=MB_FORMAT) == []


def test_write_manual_sidecar_mb(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    sidecar = write_manual_sidecar(img, ["outdoor", "single"], fmt=MB_FORMAT)
    assert sidecar.name == "photo.labels.mb"
    text = sidecar.read_text()
    assert "<<<" in text
    assert "outdoor; single" in text
    # Round-trips
    assert load_manual_sidecar(str(img), fmt=MB_FORMAT) == ["outdoor", "single"]


def test_write_manual_sidecar_mb_empty_does_not_create(tmp_path):
    """Empty labels must not create a new sidecar — UI uses existence as 'touched'."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    write_manual_sidecar(img, [], fmt=MB_FORMAT)
    assert not (tmp_path / "photo.labels.mb").exists()


def test_write_manual_sidecar_mb_empty_truncates_existing(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    write_manual_sidecar(img, ["x"], fmt=MB_FORMAT)
    write_manual_sidecar(img, [], fmt=MB_FORMAT)
    sidecar = tmp_path / "photo.labels.mb"
    assert sidecar.exists()
    assert sidecar.read_text() == ""


def test_write_final_sidecar_mb(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    sidecar = write_final_sidecar(str(img), ["aircraft", "single"], fmt=MB_FORMAT)
    assert sidecar.name == "photo.mb"
    assert MB_FORMAT.decode(sidecar.read_text()) == ["aircraft", "single"]


def test_load_detected_sidecar_mb(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.detected.mb").write_text("<<< single; calm\n")
    assert load_detected_sidecar(img, fmt=MB_FORMAT) == ["single", "calm"]


def test_merge_sidecars_mb(tmp_path):
    yaml_str = """\
defaults:
  threshold: 0.3
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
"""
    tl = parse(yaml_str)
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.detected.mb").write_text("<<< single; calm\n")
    (tmp_path / "photo.labels.mb").write_text("<<< custom-tag\n")
    merged = merge_sidecars(tl, str(img), fmt=MB_FORMAT)
    assert "custom-tag" in merged
    assert "single" in merged
    assert "calm" in merged


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


# --- Suppression via -term ---

def test_assemble_suppress_detected_label():
    """-term in manual labels suppresses that term from final output."""
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.1}}
    manual = ["-calm"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "calm" not in result.final_labels
    assert "-calm" not in result.final_labels
    assert "single" in result.final_labels


def test_assemble_suppress_global_label():
    """-term can suppress a global label."""
    tl = _parse(TAXONOMY_WITH_GLOBALS)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["-aircraft"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "aircraft" not in result.final_labels
    assert "-aircraft" not in result.final_labels
    assert "mooney m20" in result.final_labels


def test_assemble_suppress_with_additive():
    """Suppression and additive labels coexist."""
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}, "mood": {"calm": 0.5, "energetic": 0.4}}
    manual = ["custom tag", "-calm"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "custom tag" in result.final_labels
    assert "calm" not in result.final_labels
    assert "-calm" not in result.final_labels
    assert "energetic" in result.final_labels


def test_assemble_suppress_nonexistent_is_harmless():
    """-term for a term that doesn't exist is a no-op."""
    tl = _parse(TAXONOMY)
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["-nonexistent"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "-nonexistent" not in result.final_labels
    assert "single" in result.final_labels


def test_assemble_exclusive_manual_displaces_detected():
    """In exactly-one category, manual label displaces detected sibling."""
    tl = _parse(TAXONOMY)
    # Detection would pick "single" (highest score), but manual says "group"
    scores = {"count": {"single": 0.9, "group": 0.1}}
    manual = ["group"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "group" in result.final_labels
    assert "single" not in result.final_labels


def test_assemble_zero_or_one_manual_displaces_detected():
    """In zero-or-one category, manual label displaces detected sibling."""
    tl = _parse(TAXONOMY)
    scores = {"setting": {"indoor": 0.6, "outdoor": 0.3}}
    manual = ["outdoor"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "outdoor" in result.final_labels
    assert "indoor" not in result.final_labels


def test_assemble_zero_or_more_manual_does_not_displace():
    """In zero-or-more category, manual label does NOT displace siblings."""
    tl = _parse(TAXONOMY)
    scores = {"mood": {"calm": 0.5, "energetic": 0.4, "tense": 0.1}}
    manual = ["calm"]
    result = assemble_final_labels(tl, "img.jpg", scores, manual_labels=manual)
    assert "calm" in result.final_labels
    assert "energetic" in result.final_labels


# --- Open-term category assembly ---

def test_assemble_open_exclusive_manual_displaces_detected_open_value():
    """An exactly-one open category: adding a manual open value displaces a detected one."""
    tl = _parse("""\
defaults:
  threshold: 0.3
categories:
  - name: color
    mode: exactly-one
    open: true
    term_prefix: "color-"
""")
    # VLM detected color-red; user manually sets color-teal — the detected
    # open value should be suppressed.
    vlm_labels = {"color": ["color-red"]}
    result = assemble_final_labels(
        tl, "img.jpg", scores={}, manual_labels=["color-teal"], vlm_labels=vlm_labels
    )
    assert "color-teal" in result.final_labels
    assert "color-red" not in result.final_labels


def test_assemble_open_closed_term_wins_over_open_assignment():
    """color-red is closed in cat A; cat B has open prefix color-. A wins."""
    tl = _parse("""\
defaults:
  threshold: 0.3
categories:
  - name: prominent
    mode: zero-or-one
    terms:
      - term: color-red
  - name: palette
    mode: exactly-one
    open: true
    term_prefix: "color-"
""")
    # Manual color-red should displace a detected color-red? No — only one
    # copy. But the key assertion: color-red belongs to 'prominent', not
    # 'palette'. So if 'palette' also has a detected color-teal, manual
    # color-red does not displace color-teal (different categories).
    vlm_labels = {"prominent": [], "palette": ["color-teal"]}
    result = assemble_final_labels(
        tl, "img.jpg", scores={}, manual_labels=["color-red"], vlm_labels=vlm_labels
    )
    assert "color-red" in result.final_labels
    assert "color-teal" in result.final_labels


def test_assemble_open_zero_or_more_does_not_displace():
    tl = _parse("""\
defaults:
  threshold: 0.3
categories:
  - name: tags
    mode: zero-or-more
    open: true
    term_prefix: "tag-"
""")
    vlm_labels = {"tags": ["tag-cat", "tag-fluffy"]}
    result = assemble_final_labels(
        tl, "img.jpg", scores={}, manual_labels=["tag-playful"], vlm_labels=vlm_labels
    )
    assert "tag-cat" in result.final_labels
    assert "tag-fluffy" in result.final_labels
    assert "tag-playful" in result.final_labels
