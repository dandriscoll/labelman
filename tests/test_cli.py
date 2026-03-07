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
