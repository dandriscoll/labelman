"""Tests for labelman.yaml parsing."""

import pytest

from labelman.schema import CategoryMode, ParseError, parse


# --- Valid parsing ---

def test_parse_minimal():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: object
""")
    assert len(tl.categories) == 1
    assert tl.defaults.threshold == 0.3
    assert tl.categories[0].name == "subject"
    assert tl.categories[0].mode == CategoryMode.EXACTLY_ONE
    assert len(tl.categories[0].terms) == 2


def test_parse_full():
    tl = parse("""\
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
      - term: tense
        threshold: 0.25
""")
    assert len(tl.categories) == 3
    assert tl.categories[1].threshold == 0.45
    assert tl.categories[2].terms[1].threshold == 0.25


def test_parse_with_integrations():
    tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.blip is not None
    assert tl.integrations.blip.endpoint == "http://localhost:8080/caption"
    assert tl.integrations.clip is not None
    assert tl.integrations.clip.endpoint == "http://localhost:8081/classify"


def test_parse_integration_with_script():
    tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  clip:
    script: /usr/local/bin/my-clip.sh
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.clip is not None
    assert tl.integrations.clip.script == "/usr/local/bin/my-clip.sh"
    assert tl.integrations.clip.endpoint is None


def test_parse_integration_script_overrides_endpoint():
    tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  clip:
    endpoint: http://localhost:8081/classify
    script: /usr/local/bin/my-clip.sh
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.clip.script == "/usr/local/bin/my-clip.sh"
    assert tl.integrations.clip.endpoint == "http://localhost:8081/classify"


def test_parse_integrations_optional():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.blip is None
    assert tl.integrations.clip is None


def test_parse_integration_missing_endpoint_and_script():
    with pytest.raises(ParseError, match="must specify 'endpoint' or 'script'"):
        parse("""\
defaults:
  threshold: 0.3
integrations:
  clip: {}
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")


def test_parse_from_file(tmp_path):
    f = tmp_path / "labelman.yaml"
    f.write_text("""\
defaults:
  threshold: 0.5
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    tl = parse(f)
    assert tl.defaults.threshold == 0.5


# --- Structural errors ---

def test_parse_empty_file():
    with pytest.raises(ParseError, match="Top level must be a YAML mapping"):
        parse("")


def test_parse_missing_defaults():
    with pytest.raises(ParseError, match="Missing required key: 'defaults'"):
        parse("categories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - term: x\n")


def test_parse_missing_threshold():
    with pytest.raises(ParseError, match="missing required key: 'threshold'"):
        parse("defaults: {}\ncategories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - term: x\n")


def test_parse_missing_categories():
    with pytest.raises(ParseError, match="Missing required key: 'categories'"):
        parse("defaults:\n  threshold: 0.3\n")


def test_parse_empty_categories():
    with pytest.raises(ParseError, match="must not be empty"):
        parse("defaults:\n  threshold: 0.3\ncategories: []\n")


def test_parse_missing_category_name():
    with pytest.raises(ParseError, match="missing required key 'name'"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - mode: exactly-one\n    terms:\n      - term: x\n")


def test_parse_missing_category_mode():
    with pytest.raises(ParseError, match="missing required key 'mode'"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    terms:\n      - term: x\n")


def test_parse_invalid_mode():
    with pytest.raises(ParseError, match="invalid mode 'many'"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: many\n    terms:\n      - term: x\n")


def test_parse_missing_terms():
    with pytest.raises(ParseError, match="missing required key 'terms'"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: exactly-one\n")


def test_parse_empty_terms():
    with pytest.raises(ParseError, match="must not be empty"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: exactly-one\n    terms: []\n")


def test_parse_missing_term_name():
    with pytest.raises(ParseError, match="missing required key 'term'"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - threshold: 0.5\n")


def test_parse_threshold_out_of_range_global():
    with pytest.raises(ParseError, match="between 0.0 and 1.0"):
        parse("defaults:\n  threshold: 1.5\ncategories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - term: x\n")


def test_parse_threshold_out_of_range_negative():
    with pytest.raises(ParseError, match="between 0.0 and 1.0"):
        parse("defaults:\n  threshold: -0.1\ncategories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - term: x\n")


def test_parse_threshold_out_of_range_category():
    with pytest.raises(ParseError, match="between 0.0 and 1.0"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: exactly-one\n    threshold: 2.0\n    terms:\n      - term: x\n")


def test_parse_threshold_out_of_range_term():
    with pytest.raises(ParseError, match="between 0.0 and 1.0"):
        parse("defaults:\n  threshold: 0.3\ncategories:\n  - name: a\n    mode: exactly-one\n    terms:\n      - term: x\n        threshold: -1.0\n")


# --- Uniqueness ---

def test_parse_duplicate_category_names():
    with pytest.raises(ParseError, match="duplicate category name 'a'"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
  - name: a
    mode: zero-or-more
    terms:
      - term: y
""")


def test_parse_duplicate_term_names():
    with pytest.raises(ParseError, match="duplicate term 'x'"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: x
      - term: x
""")


def test_parse_same_term_across_categories_allowed():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: exactly-one
    terms:
      - term: shared
      - term: other
  - name: b
    mode: zero-or-more
    terms:
      - term: shared
""")
    assert tl.categories[0].terms[0].term == "shared"
    assert tl.categories[1].terms[0].term == "shared"


def test_parse_multiple_errors_reported():
    """Parser should report all errors, not just the first."""
    with pytest.raises(ParseError) as exc_info:
        parse("""\
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
    errors = exc_info.value.errors
    assert len(errors) >= 2


# --- Global terms ---

def test_parse_global_terms():
    tl = parse("""\
defaults:
  threshold: 0.3
global_terms:
  - aircraft
  - mooney m20
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.global_terms == ["aircraft", "mooney m20"]


def test_parse_global_terms_optional():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.global_terms == []


def test_parse_global_terms_empty_list():
    tl = parse("""\
defaults:
  threshold: 0.3
global_terms: []
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.global_terms == []


def test_parse_global_terms_not_a_list():
    with pytest.raises(ParseError, match="must be a list"):
        parse("""\
defaults:
  threshold: 0.3
global_terms: "aircraft"
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")


def test_parse_global_terms_non_string_item():
    with pytest.raises(ParseError, match="must be a string"):
        parse("""\
defaults:
  threshold: 0.3
global_terms:
  - 123
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")


def test_parse_global_terms_whitespace_trimmed():
    tl = parse("""\
defaults:
  threshold: 0.3
global_terms:
  - "  aircraft  "
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.global_terms == ["aircraft"]
