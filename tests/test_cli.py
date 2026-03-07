"""Tests for the CLI."""

import json

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
