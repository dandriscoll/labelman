"""Tests for the check command."""

from labelman.check import check


def test_check_valid_file():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: y
""")
    assert result.ok
    assert result.errors == []
    assert result.num_categories == 1
    assert result.num_terms == 2


def test_check_invalid_yaml():
    result = check("{{invalid yaml")
    assert not result.ok
    assert len(result.errors) > 0


def test_check_schema_violations():
    result = check("defaults: {}\ncategories: []\n")
    assert not result.ok
    assert len(result.errors) >= 2


def test_check_duplicate_categories():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: y
  - name: a
    mode: zero-or-more
    terms:
      - term: z
""")
    assert not result.ok
    assert any("duplicate" in e for e in result.errors)


def test_check_duplicate_terms():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: x
""")
    assert not result.ok
    assert any("duplicate" in e for e in result.errors)


def test_check_single_term_exactly_one_warning():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: only
""")
    assert result.ok
    assert any("exactly-one" in w and "1 term" in w for w in result.warnings)


def test_check_degenerate_threshold_zero():
    result = check("""\
defaults:
  threshold: 0.0
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert result.ok
    assert any("0.0" in w for w in result.warnings)


def test_check_degenerate_threshold_one():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    threshold: 1.0
    terms:
      - term: x
""")
    assert result.ok
    assert any("1.0" in w for w in result.warnings)


def test_check_summary_counts():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: y
  - name: b
    mode: zero-or-more
    terms:
      - term: z
""")
    assert result.ok
    assert result.num_categories == 2
    assert result.num_terms == 3


def test_check_multiple_errors():
    result = check("""\
defaults:
  threshold: 0.3
categories:
  - mode: exactly-one
    terms:
      - term: x
  - name: b
    terms:
      - term: y
""")
    assert not result.ok
    assert len(result.errors) >= 2
