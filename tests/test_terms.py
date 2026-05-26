"""Tests for `labelman terms` and the corpus frequency counter."""

from pathlib import Path

from labelman.build import count_terms, render_terms
from labelman.cli import main as cli_main


def _img(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8")
    return p


def test_count_terms_across_files(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    _img(tmp_path, "c.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single")
    (tmp_path / "b.txt").write_text("outdoor, calm")
    (tmp_path / "c.txt").write_text("outdoor")
    counts, n = count_terms(tmp_path)
    assert counts["outdoor"] == 3
    assert counts["single"] == 1
    assert counts["calm"] == 1
    assert n == 3


def test_count_terms_mixed_txt_and_mb(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single")
    (tmp_path / "b.mb").write_text("<<< outdoor; calm\n")
    counts, n = count_terms(tmp_path)
    assert counts["outdoor"] == 2
    assert counts["single"] == 1
    assert counts["calm"] == 1
    assert n == 2


def test_count_terms_excludes_intermediate_and_suppressions(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, -unwanted")
    (tmp_path / "a.labels.txt").write_text("manual-only")
    (tmp_path / "a.detected.txt").write_text("detected-only")
    counts, n = count_terms(tmp_path)
    assert counts == {"outdoor": 1}
    assert n == 1


def test_render_terms_sorted_by_count_then_label():
    from collections import Counter
    counts = Counter({"foo": 102, "bar": 95, "baz": 95})
    out = render_terms(counts)
    # Count descending; ties broken alphabetically (bar before baz).
    assert out == "102 foo\n95 bar\n95 baz\n"


def test_render_terms_empty():
    from collections import Counter
    assert render_terms(Counter()) == ""


# --- CLI integration ---

def test_cli_terms_default_output(tmp_path, capsys):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("outdoor, single")
    (tmp_path / "b.txt").write_text("outdoor")
    rc = cli_main(["terms", "--images", str(tmp_path)])
    assert rc == 0
    out_file = tmp_path / "terms.txt"
    assert out_file.is_file()
    assert out_file.read_text() == "2 outdoor\n1 single\n"
    summary = capsys.readouterr().out
    assert "2 unique label(s)" in summary
    assert "2 sidecar file(s)" in summary


def test_cli_terms_explicit_output(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    dest = tmp_path / "sub" / "freq.txt"
    rc = cli_main(["terms", "--images", str(tmp_path), "--output", str(dest)])
    assert rc == 0
    assert dest.read_text() == "1 outdoor\n"
    # Default file is not created when --output is given.
    assert not (tmp_path / "terms.txt").exists()


def test_cli_terms_empty_corpus_writes_empty_file(tmp_path, capsys):
    _img(tmp_path, "a.jpg")  # no sidecars
    rc = cli_main(["terms", "--images", str(tmp_path)])
    assert rc == 0
    out_file = tmp_path / "terms.txt"
    assert out_file.is_file()
    assert out_file.read_text() == ""
    assert "0 unique label(s)" in capsys.readouterr().out


def test_cli_terms_missing_dir(tmp_path, capsys):
    rc = cli_main(["terms", "--images", str(tmp_path / "nope")])
    assert rc == 1
    assert "not a directory" in capsys.readouterr().err
