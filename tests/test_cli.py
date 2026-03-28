"""Tests for the CLI."""

import json
from unittest.mock import patch

from labelman.cli import main


def test_cli_no_command(capsys):
    result = main([])
    assert result == 1


def test_cli_check_valid(tmp_path, capsys):
    f = tmp_path / "labelman.yaml"
    f.write_text("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: y
""")
    result = main(["check", "--config", str(f)])
    assert result == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "1 categories" in captured.out
    assert "2 terms" in captured.out


def test_cli_check_invalid(tmp_path, capsys):
    f = tmp_path / "labelman.yaml"
    f.write_text("defaults: {}\ncategories: []\n")
    result = main(["check", "--config", str(f)])
    assert result == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_cli_check_missing_file(capsys):
    result = main(["check", "--config", "/nonexistent/labelman.yaml"])
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_check_with_warnings(tmp_path, capsys):
    f = tmp_path / "labelman.yaml"
    f.write_text("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: only
""")
    result = main(["check", "--config", str(f)])
    assert result == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


def test_cli_init(tmp_path, capsys):
    target = tmp_path / "project"
    result = main(["init", "--dir", str(target)])
    assert result == 0
    assert (target / "labelman.yaml").exists()


def test_cli_init_refuses_overwrite(tmp_path, capsys):
    (tmp_path / "labelman.yaml").write_text("existing")
    result = main(["init", "--dir", str(tmp_path)])
    assert result == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err


def test_cli_init_force_overwrite(tmp_path, capsys):
    (tmp_path / "labelman.yaml").write_text("existing")
    result = main(["init", "--dir", str(tmp_path), "--force"])
    assert result == 0


def test_cli_init_output_passes_check(tmp_path):
    """The starter labelman.yaml from init should pass check."""
    main(["init", "--dir", str(tmp_path)])
    result = main(["check", "--config", str(tmp_path / "labelman.yaml")])
    assert result == 0


def test_cli_init_includes_integrations(tmp_path):
    main(["init", "--dir", str(tmp_path)])
    content = (tmp_path / "labelman.yaml").read_text()
    assert "integrations:" in content
    assert "blip:" in content
    assert "clip:" in content
    assert "endpoint:" in content


def test_cli_init_demonstrates_all_modes(tmp_path):
    """The starter file should show all three category modes."""
    from labelman.schema import CategoryMode, parse

    main(["init", "--dir", str(tmp_path)])
    tl = parse(tmp_path / "labelman.yaml")
    modes = {c.mode for c in tl.categories}
    assert CategoryMode.EXACTLY_ONE in modes
    assert CategoryMode.ZERO_OR_ONE in modes
    assert CategoryMode.ZERO_OR_MORE in modes


def test_cli_init_demonstrates_threshold_overrides(tmp_path):
    """The starter file should show per-category and per-term threshold overrides."""
    from labelman.schema import parse

    main(["init", "--dir", str(tmp_path)])
    tl = parse(tmp_path / "labelman.yaml")
    # At least one category with a threshold override
    cat_overrides = [c for c in tl.categories if c.threshold is not None]
    assert len(cat_overrides) >= 1
    # At least one term with a threshold override
    term_overrides = [
        t for c in tl.categories for t in c.terms if t.threshold is not None
    ]
    assert len(term_overrides) >= 1


def test_cli_init_has_comments(tmp_path):
    """The starter file should include explanatory comments."""
    main(["init", "--dir", str(tmp_path)])
    content = (tmp_path / "labelman.yaml").read_text()
    assert "exactly-one" in content
    assert "zero-or-one" in content
    assert "zero-or-more" in content
    assert "# " in content  # has comments
    assert "script:" in content  # shows custom script option


LABEL_CONFIG = """\
defaults:
  threshold: 0.3
integrations:
  clip:
    endpoint: http://localhost:8081/classify
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: tense
"""


def _make_label_workspace(tmp_path, config=LABEL_CONFIG, image_names=None):
    """Set up a workspace with config and dummy images."""
    config_path = tmp_path / "labelman.yaml"
    config_path.write_text(config)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    if image_names is None:
        image_names = ["img1.jpg", "img2.jpg"]
    for name in image_names:
        (images_dir / name).write_text("")
    return config_path, images_dir


def _mock_clip_for_label(term_list, image_paths):
    """Mock run_clip returning deterministic scores."""
    results = []
    for i, path in enumerate(image_paths):
        if i % 2 == 0:
            scores = {"person": 0.8, "animal": 0.1, "calm": 0.5, "tense": 0.2}
        else:
            scores = {"person": 0.2, "animal": 0.7, "calm": 0.1, "tense": 0.4}
        results.append({"image": path, "scores": scores})
    return results


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_produces_outputs(mock_clip, tmp_path, capsys):
    config_path, images_dir = _make_label_workspace(tmp_path)
    result = main(["label", "--images", str(images_dir), "--config", str(config_path)])
    assert result == 0
    captured = capsys.readouterr()
    assert "Done: 2 images labeled" in captured.out

    # Check sidecar files
    assert (images_dir / "img1.detected.txt").exists()
    assert (images_dir / "img2.detected.txt").exists()
    # Check CSV
    assert (images_dir / "labels.csv").exists()
    # Check report
    assert (images_dir / "report.html").exists()


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_sidecar_content(mock_clip, tmp_path):
    config_path, images_dir = _make_label_workspace(tmp_path)
    main(["label", "--images", str(images_dir), "--config", str(config_path)])
    content = (images_dir / "img1.detected.txt").read_text()
    assert "person" in content
    assert "calm" in content


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_output_dir(mock_clip, tmp_path):
    config_path, images_dir = _make_label_workspace(tmp_path)
    out_dir = tmp_path / "output"
    main(["label", "--images", str(images_dir), "--config", str(config_path),
          "--output", str(out_dir)])
    assert (out_dir / "img1.detected.txt").exists()
    assert (out_dir / "labels.csv").exists()
    assert (out_dir / "report.html").exists()


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_with_manual_sidecar(mock_clip, tmp_path):
    config_path, images_dir = _make_label_workspace(tmp_path)
    # Create a manual label sidecar
    (images_dir / "img1.labels.txt").write_text("custom tag\n")
    main(["label", "--images", str(images_dir), "--config", str(config_path)])
    # .detected.txt should only have CLIP-detected terms, not manual labels
    detected_content = (images_dir / "img1.detected.txt").read_text()
    assert "custom tag" not in detected_content
    assert "person" in detected_content
    # After apply, .txt should have manual + detected + globals
    main(["apply", "--images", str(images_dir), "--config", str(config_path)])
    final_content = (images_dir / "img1.txt").read_text()
    assert "custom tag" in final_content
    assert "person" in final_content


def test_cli_label_missing_config(capsys):
    result = main(["label", "--config", "/nonexistent/labelman.yaml"])
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_label_no_images(tmp_path, capsys):
    config_path = tmp_path / "labelman.yaml"
    config_path.write_text(LABEL_CONFIG)
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = main(["label", "--images", str(empty_dir), "--config", str(config_path)])
    assert result == 1
    captured = capsys.readouterr()
    assert "no images" in captured.err


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_progress_output(mock_clip, tmp_path, capsys):
    config_path, images_dir = _make_label_workspace(tmp_path)
    main(["label", "--images", str(images_dir), "--config", str(config_path)])
    captured = capsys.readouterr()
    assert "Labeling 2 images" in captured.out
    assert "[1/2] img1.jpg" in captured.out
    assert "[2/2] img2.jpg" in captured.out
    assert "labels.csv" in captured.out
    assert "report.html" in captured.out


@patch("labelman.cli.run_clip", side_effect=_mock_clip_for_label)
def test_cli_label_quiet(mock_clip, tmp_path, capsys):
    config_path, images_dir = _make_label_workspace(tmp_path)
    result = main(["label", "--images", str(images_dir), "--config", str(config_path), "--quiet"])
    assert result == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    # Outputs should still be written
    assert (images_dir / "img1.detected.txt").exists()
    assert (images_dir / "labels.csv").exists()


def test_cli_label_no_terms(tmp_path, capsys):
    config_path = tmp_path / "labelman.yaml"
    config_path.write_text("""\
defaults:
  threshold: 0.3
categories:
  - name: color
    mode: zero-or-one
    question: "What color?"
""")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img.jpg").write_text("")
    result = main(["label", "--images", str(images_dir), "--config", str(config_path)])
    assert result == 1
    captured = capsys.readouterr()
    assert "no terms" in captured.err


def test_cli_apply(tmp_path, capsys):
    config_path = tmp_path / "labelman.yaml"
    config_path.write_text("""\
defaults:
  threshold: 0.3
global_terms:
  - photo
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
""")
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_text("")
    (images_dir / "img1.detected.txt").write_text("person")
    (images_dir / "img1.labels.txt").write_text("custom tag")

    result = main(["apply", "--images", str(images_dir), "--config", str(config_path)])
    assert result == 0
    captured = capsys.readouterr()
    assert "Applied 1 image(s)" in captured.out

    final = (images_dir / "img1.txt").read_text()
    assert "photo" in final
    assert "custom tag" in final
    assert "person" in final


def test_cli_descriptor_blip(capsys):
    result = main(["descriptor", "blip"])
    assert result == 0
    captured = capsys.readouterr()
    desc = json.loads(captured.out)
    assert desc["name"] == "labelman-blip"
    assert "inputs" in desc


def test_cli_descriptor_clip(capsys):
    result = main(["descriptor", "clip"])
    assert result == 0
    captured = capsys.readouterr()
    desc = json.loads(captured.out)
    assert desc["name"] == "labelman-clip"
    assert "inputs" in desc
