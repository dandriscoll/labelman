"""Tests for the rename command."""

from pathlib import Path

from labelman.cli import main
from labelman.rename import rename_term


CONFIG = """\
defaults:
  threshold: 0.3
global_terms:
  - aircraft
  - mooney m20
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


def _setup(tmp_path, config=CONFIG):
    """Create config and return (config_path, base_dir).

    base_dir is the config's parent — where sidecars should live.
    """
    config_path = tmp_path / "labelman.yaml"
    config_path.write_text(config)
    return config_path, tmp_path


def test_rename_in_yaml(tmp_path):
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="calm", new="relaxed")
    assert result.error is None
    assert str(config_path) in result.changes
    text = config_path.read_text()
    assert "relaxed" in text
    assert "calm" not in text.split("relaxed")[0].split("mood")[1]  # calm replaced in mood category


def test_rename_global_term(tmp_path):
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="mooney m20", new="mooney m20j")
    assert result.error is None
    text = config_path.read_text()
    assert "mooney m20j" in text


def test_rename_preserves_yaml_formatting(tmp_path):
    config_path, base = _setup(tmp_path)
    original = config_path.read_text()
    rename_term(config_path, old="calm", new="relaxed")
    updated = config_path.read_text()
    # Comments and structure should be preserved
    assert "defaults:" in updated
    assert "global_terms:" in updated
    assert "categories:" in updated
    # Line count should be the same
    assert len(original.splitlines()) == len(updated.splitlines())


def test_rename_in_sidecar_files(tmp_path):
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")
    (base / "img1.labels.txt").write_text("custom, calm")
    (base / "img1.txt").write_text("aircraft, mooney m20, custom, person, calm")

    result = rename_term(config_path, old="calm", new="relaxed")
    assert result.error is None

    assert (base / "img1.detected.txt").read_text() == "person, relaxed"
    assert (base / "img1.labels.txt").read_text() == "custom, relaxed"
    assert (base / "img1.txt").read_text() == "aircraft, mooney m20, custom, person, relaxed"


def test_rename_suppressed_term(tmp_path):
    config_path, base = _setup(tmp_path)
    (base / "img1.labels.txt").write_text("-calm, tense")

    result = rename_term(config_path, old="calm", new="relaxed")
    assert result.error is None
    assert (base / "img1.labels.txt").read_text() == "-relaxed, tense"


def test_rename_dry_run_no_changes(tmp_path):
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")
    original_yaml = config_path.read_text()
    original_sidecar = (base / "img1.detected.txt").read_text()

    result = rename_term(config_path, old="calm", new="relaxed", dry_run=True)
    assert result.error is None
    assert len(result.changes) > 0  # changes reported

    # But files are untouched
    assert config_path.read_text() == original_yaml
    assert (base / "img1.detected.txt").read_text() == original_sidecar


def test_rename_term_not_found(tmp_path):
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="nonexistent", new="whatever")
    assert result.error is not None
    assert "not found" in result.error


def test_rename_target_already_exists(tmp_path):
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="calm", new="tense")
    assert result.error is not None
    assert "already exists" in result.error


def test_rename_same_name(tmp_path):
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="calm", new="calm")
    assert result.error is not None
    assert "same" in result.error


def test_rename_no_sidecars(tmp_path):
    """When there are no sidecar files, only the YAML is changed."""
    config_path, base = _setup(tmp_path)
    result = rename_term(config_path, old="calm", new="relaxed")
    assert result.error is None
    assert len(result.changes) == 1
    assert "labelman.yaml" in result.changes[0]


def test_rename_unaffected_sidecar_not_listed(tmp_path):
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")
    (base / "img2.detected.txt").write_text("person, tense")

    result = rename_term(config_path, old="calm", new="relaxed")
    # img2 should not be in changes since it doesn't contain "calm"
    change_names = [Path(c).name for c in result.changes]
    assert "img1.detected.txt" in change_names
    assert "img2.detected.txt" not in change_names


def test_rename_cli(tmp_path, capsys):
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")

    result = main([
        "rename", "--old", "calm", "--new", "relaxed",
        "--config", str(config_path),
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert "Renamed" in captured.out
    assert "relaxed" in captured.out


def test_rename_cli_dry_run(tmp_path, capsys):
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")

    result = main([
        "rename", "--old", "calm", "--new", "relaxed",
        "--config", str(config_path),
        "--dry-run",
    ])
    assert result == 0
    captured = capsys.readouterr()
    assert "[dry run]" in captured.out
    # File unchanged
    assert (base / "img1.detected.txt").read_text() == "person, calm"


def test_rename_cli_error(tmp_path, capsys):
    config_path, base = _setup(tmp_path)
    result = main([
        "rename", "--old", "nope", "--new", "whatever",
        "--config", str(config_path),
    ])
    assert result == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_rename_to_multi_word(tmp_path):
    """Rename from a single word to a multi-word term with spaces."""
    config_path, base = _setup(tmp_path)
    (base / "img1.detected.txt").write_text("person, calm")
    (base / "img1.txt").write_text("aircraft, person, calm")

    result = rename_term(config_path, old="calm", new="calm and peaceful")
    assert result.error is None

    text = config_path.read_text()
    assert "calm and peaceful" in text

    assert (base / "img1.detected.txt").read_text() == "person, calm and peaceful"
    assert (base / "img1.txt").read_text() == "aircraft, person, calm and peaceful"
