"""Tests for `labelman build`."""

from pathlib import Path

import pytest

from labelman.build import build_yaml, collect_terms, render_yaml
from labelman.cli import main as cli_main
from labelman.errors import LabelmanError
from labelman.schema import parse


def _img(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8")
    return p


def test_collect_terms_from_txt(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single, calm")
    (tmp_path / "b.txt").write_text("indoor, single")
    terms, n = collect_terms(tmp_path)
    assert terms == ["calm", "indoor", "outdoor", "single"]
    assert n == 2


def test_collect_terms_from_mb(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.mb").write_text("<<< outdoor; single; calm\n")
    terms, n = collect_terms(tmp_path)
    assert terms == ["calm", "outdoor", "single"]
    assert n == 1


def test_collect_terms_mixed_txt_and_mb(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single")
    (tmp_path / "b.mb").write_text("<<< indoor; calm\n")
    terms, n = collect_terms(tmp_path)
    assert terms == ["calm", "indoor", "outdoor", "single"]
    assert n == 2


def test_collect_terms_excludes_intermediate_sidecars(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.labels.txt").write_text("manual-only")
    (tmp_path / "a.detected.txt").write_text("detected-only")
    (tmp_path / "a.labels.mb").write_text("<<< mb-manual\n")
    (tmp_path / "a.detected.mb").write_text("<<< mb-detected\n")
    terms, n = collect_terms(tmp_path)
    assert terms == []
    assert n == 0


def test_collect_terms_filters_suppression_directives(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, -unwanted, calm")
    terms, _ = collect_terms(tmp_path)
    assert "outdoor" in terms
    assert "calm" in terms
    assert "-unwanted" not in terms
    assert "unwanted" not in terms


def test_collect_terms_dedupes_across_files(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("single, calm")
    (tmp_path / "b.txt").write_text("single, energetic")
    terms, _ = collect_terms(tmp_path)
    assert terms == ["calm", "energetic", "single"]


def test_render_yaml_each_term_zero_or_one(tmp_path):
    yaml_str = render_yaml(["calm", "outdoor", "single"])
    tl = parse(yaml_str)
    names = [c.name for c in tl.categories]
    assert names == ["calm", "outdoor", "single"]
    for cat in tl.categories:
        assert cat.mode.value == "zero-or-one"
        assert [t.term for t in cat.terms] == [cat.name]


def test_render_yaml_comments_at_top(tmp_path):
    yaml_str = render_yaml(["outdoor"])
    lines = yaml_str.splitlines()
    # Find the first non-comment, non-blank line; everything before must be
    # a comment or blank.
    for i, line in enumerate(lines):
        s = line.strip()
        if s and not s.startswith("#"):
            first_data = i
            break
    for line in lines[:first_data]:
        s = line.strip()
        assert s == "" or s.startswith("#")
    # And no comment appears interleaved with category data.
    for line in lines[first_data:]:
        assert not line.strip().startswith("#")


def test_render_yaml_quotes_terms_with_special_chars(tmp_path):
    yaml_str = render_yaml(["a, comma", "plain"])
    # round-trips
    tl = parse(yaml_str)
    names = [c.name for c in tl.categories]
    assert "a, comma" in names
    assert "plain" in names


def test_build_yaml_passes_check(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single")
    text, terms, _ = build_yaml(tmp_path)
    # Must parse cleanly.
    tl = parse(text)
    assert {c.name for c in tl.categories} == set(terms)


# --- CLI integration ---

def test_cli_build_writes_yaml(tmp_path, capsys):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, calm")
    rc = cli_main(["build", "--images", str(tmp_path)])
    assert rc == 0
    cfg = tmp_path / "labelman.yaml"
    assert cfg.is_file()
    tl = parse(cfg)
    assert {c.name for c in tl.categories} == {"outdoor", "calm"}


def test_cli_build_refuses_existing_without_force(tmp_path, capsys):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    (tmp_path / "labelman.yaml").write_text("# pre-existing\n")
    rc = cli_main(["build", "--images", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    # File untouched.
    assert (tmp_path / "labelman.yaml").read_text() == "# pre-existing\n"


def test_cli_build_force_overwrites(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    (tmp_path / "labelman.yaml").write_text("# pre-existing\n")
    rc = cli_main(["build", "--images", str(tmp_path), "--force"])
    assert rc == 0
    text = (tmp_path / "labelman.yaml").read_text()
    assert "outdoor" in text
    assert "# pre-existing" not in text


def test_cli_build_errors_on_no_terms(tmp_path, capsys):
    _img(tmp_path, "a.jpg")
    # Only intermediate sidecars exist.
    (tmp_path / "a.labels.txt").write_text("manual-only")
    rc = cli_main(["build", "--images", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no labels found" in err


def test_cli_build_errors_on_missing_dir(tmp_path, capsys):
    rc = cli_main(["build", "--images", str(tmp_path / "nope")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a directory" in err
