"""Tests for `labelman terms` and the corpus frequency counter."""

from pathlib import Path

from labelman.build import count_terms, render_terms, render_terms_markdown
from labelman.cli import main as cli_main
from labelman.schema import parse


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


# --- Markdown rendering ---

_CONFIG = """\
defaults:
  threshold: 0.3
categories:
  - name: lighting
    mode: zero-or-one
    terms:
      - term: natural
      - term: studio
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
"""


def test_render_terms_markdown_no_config():
    from collections import Counter
    counts = Counter({"outdoor": 3, "calm": 1, "single": 2})
    # Without a term list everything falls under Uncategorized, alphabetized.
    assert render_terms_markdown(counts) == (
        "## Uncategorized\n- calm\n- outdoor\n- single\n"
    )


def test_render_terms_markdown_empty():
    from collections import Counter
    assert render_terms_markdown(Counter()) == ""
    assert render_terms_markdown(Counter(), parse(_CONFIG)) == ""


def test_render_terms_markdown_grouped_by_category():
    from collections import Counter
    term_list = parse(_CONFIG)
    counts = Counter({"natural": 5, "studio": 2, "calm": 1, "outdoor": 9})
    out = render_terms_markdown(counts, term_list)
    # Categories in config order, terms alphabetized within, unknowns last.
    assert out == (
        "## lighting\n- natural\n- studio\n\n"
        "## mood\n- calm\n\n"
        "## Uncategorized\n- outdoor\n"
    )


def test_render_terms_markdown_omits_empty_categories():
    from collections import Counter
    term_list = parse(_CONFIG)
    # No 'mood' terms in the corpus, so that section is dropped entirely.
    counts = Counter({"natural": 1})
    assert render_terms_markdown(counts, term_list) == "## lighting\n- natural\n"


def test_render_terms_markdown_open_category():
    from collections import Counter
    config = """\
defaults:
  threshold: 0.3
categories:
  - name: color
    mode: zero-or-more
    open: true
    term_prefix: "color-"
    terms: []
"""
    term_list = parse(config)
    counts = Counter({"color-red": 2, "color-teal": 1, "plain": 1})
    assert render_terms_markdown(counts, term_list) == (
        "## color\n- color-red\n- color-teal\n\n"
        "## Uncategorized\n- plain\n"
    )


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


def test_cli_terms_writes_markdown_sibling(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("natural, calm")
    (tmp_path / "b.txt").write_text("studio")
    (tmp_path / "labelman.yaml").write_text(_CONFIG)
    rc = cli_main(["terms", "--images", str(tmp_path),
                   "--config", str(tmp_path / "labelman.yaml")])
    assert rc == 0
    md = tmp_path / "terms.md"
    assert md.is_file()
    assert md.read_text() == (
        "## lighting\n- natural\n- studio\n\n"
        "## mood\n- calm\n"
    )


def test_cli_terms_markdown_without_config(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, calm")
    # Point --config at a path that does not exist: terms.md is still written,
    # everything under Uncategorized.
    rc = cli_main(["terms", "--images", str(tmp_path),
                   "--config", str(tmp_path / "missing.yaml")])
    assert rc == 0
    assert (tmp_path / "terms.md").read_text() == (
        "## Uncategorized\n- calm\n- outdoor\n"
    )


def test_cli_terms_markdown_path_follows_output(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    dest = tmp_path / "sub" / "freq.txt"
    rc = cli_main(["terms", "--images", str(tmp_path), "--output", str(dest)])
    assert rc == 0
    # The .md sibling tracks the --output base name.
    assert (tmp_path / "sub" / "freq.md").read_text() == "## Uncategorized\n- outdoor\n"


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


def test_count_terms_excludes_reserved_terms_txt(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor, calm")
    # A stale terms.txt must never be scanned as a label sidecar.
    (tmp_path / "terms.txt").write_text("2 outdoor\n1 calm\n")
    counts, n = count_terms(tmp_path)
    assert counts == {"outdoor": 1, "calm": 1}
    assert n == 1  # terms.txt not counted as a scanned sidecar


def test_cli_terms_idempotent_on_rerun(tmp_path):
    _img(tmp_path, "a.jpg")
    _img(tmp_path, "b.jpg")
    (tmp_path / "a.txt").write_text("outdoor, calm")
    (tmp_path / "b.txt").write_text("outdoor")
    cli_main(["terms", "--images", str(tmp_path)])
    first = (tmp_path / "terms.txt").read_text()
    cli_main(["terms", "--images", str(tmp_path)])
    second = (tmp_path / "terms.txt").read_text()
    assert first == second == "2 outdoor\n1 calm\n"


def test_cli_terms_custom_output_inside_dir_excluded(tmp_path):
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    dest = tmp_path / "freq.txt"
    cli_main(["terms", "--images", str(tmp_path), "--output", str(dest)])
    first = dest.read_text()
    # Re-run: the prior freq.txt lives in the scanned dir but must be excluded.
    cli_main(["terms", "--images", str(tmp_path), "--output", str(dest)])
    assert dest.read_text() == first == "1 outdoor\n"


def test_collect_terms_ignores_stale_terms_txt(tmp_path):
    # build's scanner shares the exclusion: a stale terms.txt is not a sidecar.
    from labelman.build import collect_terms
    _img(tmp_path, "a.jpg")
    (tmp_path / "a.txt").write_text("outdoor")
    (tmp_path / "terms.txt").write_text("5 outdoor\n")
    terms, n = collect_terms(tmp_path)
    assert terms == ["outdoor"]
    assert n == 1
