"""Tests for threshold inheritance."""

from labelman.schema import parse


def _make(global_t, cat_t=None, term_t=None):
    cat_line = f"\n    threshold: {cat_t}" if cat_t is not None else ""
    term_line = f"\n        threshold: {term_t}" if term_t is not None else ""
    return parse(f"""\
defaults:
  threshold: {global_t}
categories:
  - name: cat
    mode: zero-or-more{cat_line}
    terms:
      - term: t{term_line}
""")


def test_threshold_global_only():
    tl = _make(0.3)
    cat = tl.categories[0]
    assert tl.effective_threshold(cat, cat.terms[0]) == 0.3


def test_threshold_category_override():
    tl = _make(0.3, cat_t=0.5)
    cat = tl.categories[0]
    assert tl.effective_threshold(cat, cat.terms[0]) == 0.5


def test_threshold_term_override():
    tl = _make(0.3, cat_t=0.5, term_t=0.1)
    cat = tl.categories[0]
    assert tl.effective_threshold(cat, cat.terms[0]) == 0.1


def test_threshold_term_overrides_global():
    tl = _make(0.3, term_t=0.7)
    cat = tl.categories[0]
    assert tl.effective_threshold(cat, cat.terms[0]) == 0.7


def test_threshold_mixed():
    tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: a
    mode: zero-or-more
    threshold: 0.5
    terms:
      - term: t1
      - term: t2
        threshold: 0.1
  - name: b
    mode: zero-or-more
    terms:
      - term: t3
      - term: t4
        threshold: 0.8
""")
    a = tl.categories[0]
    b = tl.categories[1]
    assert tl.effective_threshold(a, a.terms[0]) == 0.5  # category override
    assert tl.effective_threshold(a, a.terms[1]) == 0.1  # term override
    assert tl.effective_threshold(b, b.terms[0]) == 0.3  # global
    assert tl.effective_threshold(b, b.terms[1]) == 0.8  # term override
