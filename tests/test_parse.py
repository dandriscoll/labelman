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
    with pytest.raises(ParseError, match="top level must be a YAML mapping"):
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


def test_parse_category_question():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: lighting
    mode: zero-or-one
    question: "What type of lighting is in this image?"
    terms:
      - term: natural
      - term: studio
""")
    assert tl.categories[0].question == "What type of lighting is in this image?"


def test_parse_category_question_optional():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
""")
    assert tl.categories[0].question is None


def test_parse_category_question_empty():
    with pytest.raises(ParseError, match="must not be empty"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    question: ""
    terms:
      - term: person
      - term: animal
""")


def test_parse_category_question_no_terms():
    """A category with a question but no terms should parse successfully."""
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: lighting
    mode: zero-or-one
    question: "What type of lighting is in this image?"
""")
    assert tl.categories[0].name == "lighting"
    assert tl.categories[0].question == "What type of lighting is in this image?"
    assert tl.categories[0].terms == []


def test_parse_category_question_empty_terms():
    """A category with a question and empty terms list should parse successfully."""
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: lighting
    mode: zero-or-one
    question: "What type of lighting is in this image?"
    terms: []
""")
    assert tl.categories[0].terms == []


def test_parse_category_no_terms_no_question():
    """A category without terms AND without a question should fail."""
    with pytest.raises(ParseError, match="missing required key 'terms'"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
""")


def test_parse_llm_integration():
    tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  llm:
    endpoint: http://localhost:11434/v1/chat/completions
    model: llama3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.llm is not None
    assert tl.integrations.llm.endpoint == "http://localhost:11434/v1/chat/completions"
    assert tl.integrations.llm.model == "llama3"


def test_parse_llm_integration_no_model():
    tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  llm:
    endpoint: http://localhost:11434/v1/chat/completions
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")
    assert tl.integrations.llm is not None
    assert tl.integrations.llm.model is None


def test_parse_llm_integration_missing_endpoint():
    with pytest.raises(ParseError, match="integrations.llm.*endpoint"):
        parse("""\
defaults:
  threshold: 0.3
integrations:
  llm:
    model: llama3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
""")


def test_parse_term_ask():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: vis
    mode: zero-or-one
    terms:
      - term: front
        ask: "Is a person visible through the front window?"
      - term: side
        ask: "Is a person visible through a side window?"
""")
    assert tl.categories[0].terms[0].ask == "Is a person visible through the front window?"
    assert tl.categories[0].terms[1].ask == "Is a person visible through a side window?"


def test_parse_term_ask_string_term_has_no_ask():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - plain_string
""")
    assert tl.categories[0].terms[0].ask is None


def test_parse_category_ask():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: setting
    mode: zero-or-one
    question: "Is this indoors or outdoors?"
    ask: "Describe the setting."
    terms:
      - term: indoor
      - term: outdoor
""")
    assert tl.categories[0].question == "Is this indoors or outdoors?"
    assert tl.categories[0].ask == "Describe the setting."


def test_parse_term_ask_negative():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: prop
    mode: zero-or-one
    terms:
      - term: visible
        ask: "Is a propeller visible?"
        ask_negative: "Is the propeller hidden?"
""")
    assert tl.categories[0].terms[0].ask == "Is a propeller visible?"
    assert tl.categories[0].terms[0].ask_negative == "Is the propeller hidden?"


def test_parse_term_no_ask_negative():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    terms:
      - term: x
        ask: "Is x?"
""")
    assert tl.categories[0].terms[0].ask_negative is None


# --- Open-term categories ---

def test_parse_open_category_prefix():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: color
    mode: zero-or-more
    open: true
    term_prefix: "color-"
    terms:
      - term: color-red
      - term: color-blue
""")
    cat = tl.categories[0]
    assert cat.open is True
    assert cat.term_prefix == "color-"
    assert cat.term_suffix is None
    assert cat.matches_open_pattern("color-teal") is True
    assert cat.matches_open_pattern("color-") is False  # empty body
    assert cat.matches_open_pattern("red") is False
    assert cat.matches_open_pattern("color-red") is True


def test_parse_open_category_suffix_only():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: language
    mode: zero-or-one
    open: true
    term_suffix: "-lang"
""")
    cat = tl.categories[0]
    assert cat.matches_open_pattern("python-lang") is True
    assert cat.matches_open_pattern("lang") is False
    assert cat.matches_open_pattern("python") is False


def test_parse_open_requires_affix():
    with pytest.raises(ParseError, match="require at least one of 'term_prefix' or 'term_suffix'"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: x
    mode: zero-or-more
    open: true
""")


def test_parse_affix_without_open():
    with pytest.raises(ParseError, match="only apply when 'open: true'"):
        parse("""\
defaults:
  threshold: 0.3
categories:
  - name: x
    mode: zero-or-more
    term_prefix: "x-"
    terms:
      - term: x-a
""")


def test_parse_open_category_empty_terms_ok():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: color
    mode: zero-or-more
    open: true
    term_prefix: "color-"
""")
    assert tl.categories[0].terms == []


def test_assign_term_closed_wins_over_open():
    # "color-red" is a closed term in category A; category B has open prefix "color-".
    # Closed match wins.
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: prominent
    mode: zero-or-more
    terms:
      - term: color-red
  - name: palette
    mode: zero-or-more
    open: true
    term_prefix: "color-"
""")
    cat = tl.assign_term_to_category("color-red")
    assert cat is not None
    assert cat.name == "prominent"

    # An open-only term resolves to the open category.
    cat2 = tl.assign_term_to_category("color-teal")
    assert cat2 is not None
    assert cat2.name == "palette"

    # Unrelated term → None.
    assert tl.assign_term_to_category("banana") is None


def test_assign_longest_affix_wins_among_open():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: broad
    mode: zero-or-more
    open: true
    term_prefix: "x-"
  - name: narrow
    mode: zero-or-more
    open: true
    term_prefix: "x-y-"
""")
    cat = tl.assign_term_to_category("x-y-apple")
    assert cat is not None
    assert cat.name == "narrow"
