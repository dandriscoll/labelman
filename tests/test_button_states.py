"""Comprehensive tests for term button state cycling.

These tests simulate the exact behavior of the frontend JavaScript by
replicating the state machine logic in Python and verifying that the
API calls produce correct results on disk.

The frontend has three functions that modify currentLabels:
  doAssert():   remove -term, clearCategoryExclusives, add term
  doSuppress(): remove term, add -term
  doRestore():  remove term, remove -term

Single-image tri-state rotation:
  Exclusive + Blue:     Blue → Green → Red → Blue
  Non-exclusive + Blue: Blue → Red → Green → Blue
  Grey (any):           Grey → Green → Red → Grey

Bulk (multi-select) rotation:
  Not active/suppressed → Assert (green, siblings red in exclusive)
  Active (green) → Suppress (red)
  Suppressed + exclusive → Assert (green, siblings red)
  Suppressed + non-exclusive → Restore (back to detected/grey)
"""

import json
from http.client import HTTPConnection
from pathlib import Path

import pytest

from labelman.label import load_manual_sidecar
from labelman.web import ImageIndex, load_detected_sidecar, start_server_background


# --- Helpers ---

def _get_json(conn, path):
    conn.request("GET", path)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def _put_json(conn, path, data):
    body = json.dumps(data).encode()
    conn.request("PUT", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def _post_json(conn, path, data):
    body = json.dumps(data).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def _labels(conn, name):
    """Get the current manual + detected labels for an image."""
    _, data = _get_json(conn, f"/api/images/{name}/labels")
    return data["manual_labels"], data["detected_labels"]


def _disk_manual(dataset, name):
    return load_manual_sidecar(str(dataset / name))


def _disk_detected(dataset, name):
    return load_detected_sidecar(dataset / name)


# --- State machine: replicates the JavaScript exactly ---

def _clear_exclusives(labels, detected, term, cat_terms, mode):
    """Replicate clearCategoryExclusives from the JS."""
    if mode not in ("exactly-one", "zero-or-one"):
        return labels
    result = list(labels)
    for t in cat_terms:
        if t == term:
            continue
        if t in result:
            result.remove(t)
        if t in detected and ("-" + t) not in result:
            result.append("-" + t)
    return result


def do_assert(labels, detected, term, cat_terms, mode):
    """Replicate doAssert: remove suppression, clear exclusives, add term."""
    result = list(labels)
    sup = "-" + term
    if sup in result:
        result.remove(sup)
    result = _clear_exclusives(result, detected, term, cat_terms, mode)
    if term not in result:
        result.append(term)
    return result


def do_suppress(labels, term):
    """Replicate doSuppress: remove term, add suppression."""
    result = list(labels)
    if term in result:
        result.remove(term)
    sup = "-" + term
    if sup not in result:
        result.append(sup)
    return result


def do_restore(labels, term):
    """Replicate doRestore: remove both term and suppression."""
    result = list(labels)
    if term in result:
        result.remove(term)
    sup = "-" + term
    if sup in result:
        result.remove(sup)
    return result


def click_term(labels, detected, term, cat_terms, mode):
    """Simulate a single click on a term button.

    Returns the new labels array (what would be PUT to the server).
    """
    additive = [l for l in labels if not l.startswith("-")]
    suppressions = {l[1:] for l in labels if l.startswith("-")}
    is_exclusive = mode in ("exactly-one", "zero-or-one")
    has_manual_selection = is_exclusive and any(t in additive for t in cat_terms)

    is_manual = term in additive
    is_detected_only = term in detected and term not in additive
    is_suppressed = term in suppressions
    is_implicitly_suppressed = (not is_suppressed and not is_manual and has_manual_selection)
    was_detected = term in detected

    is_explicitly_suppressed = ("-" + term) in labels

    # Apply the same branching as the JS click handler.
    # Single-select is always correction-first:
    #   Blue (any mode): Blue → Red → Green → Blue
    #   Grey (any mode): Grey → Green → Red → Grey
    #   Implicitly suppressed: treated as grey (assert)
    if is_implicitly_suppressed:
        return do_assert(labels, detected, term, cat_terms, mode)
    elif is_explicitly_suppressed and was_detected:
        return do_assert(labels, detected, term, cat_terms, mode)
    elif is_explicitly_suppressed:
        return do_restore(labels, term)
    elif is_detected_only:
        return do_suppress(labels, term)
    elif is_manual and was_detected:
        return do_restore(labels, term)
    elif is_manual:
        return do_suppress(labels, term)
    else:
        return do_assert(labels, detected, term, cat_terms, mode)


def get_visual_state(labels, detected, term, cat_terms, mode):
    """Return the visual state of a term button.

    Returns 'red', 'green', 'blue', or 'grey'.
    Note: 'red' covers both explicit suppression (-term in labels)
    and implicit suppression (exclusive category with a sibling selected).
    """
    additive = [l for l in labels if not l.startswith("-")]
    suppressions = {l[1:] for l in labels if l.startswith("-")}
    is_exclusive = mode in ("exactly-one", "zero-or-one")
    has_manual_selection = is_exclusive and any(t in additive for t in cat_terms)

    is_manual = term in additive
    is_suppressed = term in suppressions
    is_implicitly_suppressed = not is_suppressed and not is_manual and has_manual_selection
    is_detected_only = term in detected and term not in additive

    if is_suppressed or is_implicitly_suppressed:
        return "red"
    elif is_manual:
        return "green"
    elif is_detected_only:
        return "blue"
    else:
        return "grey"


# --- Fixtures ---

@pytest.fixture
def make_dataset(tmp_path):
    """Factory to create a dataset with taxonomy, images, and sidecars."""
    def _make(categories_yaml, images=None, detected=None, manual=None, global_terms=None):
        images = images or ["img1.jpg", "img2.jpg", "img3.jpg"]
        detected = detected or {}
        manual = manual or {}
        global_terms = global_terms or []

        gt = ""
        if global_terms:
            gt = "global_terms:\n" + "".join(f"  - {t}\n" for t in global_terms)

        config = tmp_path / "labelman.yaml"
        config.write_text(f"defaults:\n  threshold: 0.3\n{gt}categories:\n{categories_yaml}")

        for name in images:
            (tmp_path / name).write_bytes(b"\xff\xd8dummy")
        for name, labels in detected.items():
            (tmp_path / name.replace(".jpg", ".detected.txt")).write_text(", ".join(labels))
        for name, labels in manual.items():
            (tmp_path / name.replace(".jpg", ".labels.txt")).write_text(", ".join(labels))

        srv, port = start_server_background(tmp_path)
        conn = HTTPConnection("127.0.0.1", port)
        return conn, srv, tmp_path

    return _make


EXACTLY_ONE_CAT = """\
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
      - term: landscape
"""

ZERO_OR_ONE_CAT = """\
  - name: lighting
    mode: zero-or-one
    terms:
      - term: natural
      - term: studio
      - term: artificial
"""

ZERO_OR_MORE_CAT = """\
  - name: style
    mode: zero-or-more
    terms:
      - term: photographic
      - term: illustration
      - term: abstract
"""

MIXED_CATS = """\
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
  - name: style
    mode: zero-or-more
    terms:
      - term: photographic
      - term: illustration
"""


# ============================================================
# Category 1: Single-select - Grey term clicks
# ============================================================

class TestSingleGreyTerm:
    """Starting from grey (no manual, no detected), click cycles: Grey → Green → Red → Grey."""

    def test_grey_to_green_exactly_one(self, make_dataset):
        conn, srv, ds = make_dataset(EXACTLY_ONE_CAT, images=["img.jpg"])
        try:
            labels = click_term([], [], "person", ["person", "animal", "landscape"], "exactly-one")
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": labels})
            manual, _ = _labels(conn, "img.jpg")
            assert "person" in manual
            assert get_visual_state(manual, [], "person", ["person", "animal", "landscape"], "exactly-one") == "green"
        finally:
            srv.shutdown()

    def test_grey_to_green_zero_or_more(self, make_dataset):
        conn, srv, ds = make_dataset(ZERO_OR_MORE_CAT, images=["img.jpg"])
        try:
            labels = click_term([], [], "photographic", ["photographic", "illustration", "abstract"], "zero-or-more")
            assert labels == ["photographic"]
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": labels})
            manual, _ = _labels(conn, "img.jpg")
            assert manual == ["photographic"]
        finally:
            srv.shutdown()

    def test_grey_green_red_grey_cycle(self, make_dataset):
        """Full cycle: Grey → Green → Red → Grey."""
        conn, srv, ds = make_dataset(ZERO_OR_MORE_CAT, images=["img.jpg"])
        try:
            terms = ["photographic", "illustration", "abstract"]
            detected = []
            mode = "zero-or-more"
            term = "photographic"

            # Grey → Green
            labels = click_term([], detected, term, terms, mode)
            assert get_visual_state(labels, detected, term, terms, mode) == "green"

            # Green → Red
            labels = click_term(labels, detected, term, terms, mode)
            assert get_visual_state(labels, detected, term, terms, mode) == "red"

            # Red → Grey
            labels = click_term(labels, detected, term, terms, mode)
            assert get_visual_state(labels, detected, term, terms, mode) == "grey"
            assert labels == []

            # Verify on disk
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": labels})
            manual, _ = _labels(conn, "img.jpg")
            assert manual == []
        finally:
            srv.shutdown()

    def test_grey_term_exclusive_suppresses_nothing(self, make_dataset):
        """Asserting grey in exclusive doesn't suppress non-detected siblings."""
        terms = ["person", "animal", "landscape"]
        labels = click_term([], [], "person", terms, "exactly-one")
        assert "person" in labels
        # No siblings are detected, so no suppressions
        assert "-animal" not in labels
        assert "-landscape" not in labels


# ============================================================
# Category 2: Single-select - Blue/detected term clicks
# ============================================================

class TestSingleDetectedTerm:
    """Starting from blue (detected but no manual), cycle depends on exclusivity."""

    def test_blue_red_green_blue_exactly_one(self, make_dataset):
        """exactly-one + detected: Blue → Red → Green → Blue."""
        terms = ["person", "animal", "landscape"]
        detected = ["person"]
        mode = "exactly-one"

        # Blue → Red
        labels = click_term([], detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "red"
        assert "-person" in labels

        # Red → Green
        labels = click_term(labels, detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "green"
        assert "person" in labels

        # Green → Blue (restore)
        labels = click_term(labels, detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "blue"
        assert "person" not in labels
        assert "-person" not in labels

    def test_blue_red_green_blue_zero_or_one(self, make_dataset):
        """zero-or-one + detected: Blue → Red → Green → Blue."""
        terms = ["natural", "studio", "artificial"]
        detected = ["natural"]
        mode = "zero-or-one"

        labels = click_term([], detected, "natural", terms, mode)
        assert get_visual_state(labels, detected, "natural", terms, mode) == "red"

        labels = click_term(labels, detected, "natural", terms, mode)
        assert get_visual_state(labels, detected, "natural", terms, mode) == "green"

        labels = click_term(labels, detected, "natural", terms, mode)
        assert get_visual_state(labels, detected, "natural", terms, mode) == "blue"

    def test_blue_red_green_blue_zero_or_more(self, make_dataset):
        """zero-or-more + detected: Blue → Red → Green → Blue."""
        terms = ["photographic", "illustration", "abstract"]
        detected = ["photographic"]
        mode = "zero-or-more"

        labels = click_term([], detected, "photographic", terms, mode)
        assert get_visual_state(labels, detected, "photographic", terms, mode) == "red"

        labels = click_term(labels, detected, "photographic", terms, mode)
        assert get_visual_state(labels, detected, "photographic", terms, mode) == "green"

        labels = click_term(labels, detected, "photographic", terms, mode)
        assert get_visual_state(labels, detected, "photographic", terms, mode) == "blue"
        assert "photographic" not in labels
        assert "-photographic" not in labels

    def test_blue_suppress_no_sibling_effect_any_mode(self, make_dataset):
        """Blue→Red (first click) never affects siblings, regardless of mode."""
        conn, srv, ds = make_dataset(
            EXACTLY_ONE_CAT, images=["img.jpg"],
            detected={"img.jpg": ["person", "animal"]},
        )
        try:
            terms = ["person", "animal", "landscape"]
            detected = ["person", "animal"]
            # Click person: blue → red (correction first)
            labels = click_term([], detected, "person", terms, "exactly-one")
            assert "-person" in labels
            # animal should be unaffected (still blue)
            assert "-animal" not in labels
            assert get_visual_state(labels, detected, "person", terms, "exactly-one") == "red"
            assert get_visual_state(labels, detected, "animal", terms, "exactly-one") == "blue"
        finally:
            srv.shutdown()

    def test_blue_to_red_to_green_suppresses_exclusive_siblings(self, make_dataset):
        """After Blue→Red→Green, asserting green DOES suppress siblings in exclusive."""
        conn, srv, ds = make_dataset(
            EXACTLY_ONE_CAT, images=["img.jpg"],
            detected={"img.jpg": ["person", "animal"]},
        )
        try:
            terms = ["person", "animal", "landscape"]
            detected = ["person", "animal"]
            # Blue → Red
            labels = click_term([], detected, "person", terms, "exactly-one")
            assert get_visual_state(labels, detected, "person", terms, "exactly-one") == "red"
            # Red → Green (assert — now siblings get suppressed)
            labels = click_term(labels, detected, "person", terms, "exactly-one")
            assert "person" in labels
            assert "-animal" in labels
            assert get_visual_state(labels, detected, "person", terms, "exactly-one") == "green"
            assert get_visual_state(labels, detected, "animal", terms, "exactly-one") == "red"

            # Verify on disk
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": labels})
            disk_detected = _disk_detected(ds, "img.jpg")
            assert "animal" not in disk_detected
        finally:
            srv.shutdown()

    def test_blue_suppress_no_sibling_effect_nonexclusive(self, make_dataset):
        """Blue→Red in non-exclusive does NOT affect siblings."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            detected={"img.jpg": ["photographic", "illustration"]},
        )
        try:
            terms = ["photographic", "illustration", "abstract"]
            detected = ["photographic", "illustration"]
            labels = click_term([], detected, "photographic", terms, "zero-or-more")
            assert "-photographic" in labels
            assert "illustration" not in labels
            assert "-illustration" not in labels
            assert get_visual_state(labels, detected, "illustration", terms, "zero-or-more") == "blue"
        finally:
            srv.shutdown()


# ============================================================
# Category 3: Single-select - Green/active term clicks
# ============================================================

class TestSingleActiveTerm:

    def test_green_was_grey_to_red(self):
        """Green (was grey) → Red."""
        terms = ["photographic", "illustration", "abstract"]
        labels = ["photographic"]
        detected = []
        mode = "zero-or-more"
        new_labels = click_term(labels, detected, "photographic", terms, mode)
        assert get_visual_state(new_labels, detected, "photographic", terms, mode) == "red"
        assert "-photographic" in new_labels

    def test_green_was_grey_full_cycle(self):
        """Grey → Green → Red → Grey."""
        terms = ["person", "animal"]
        detected = []
        mode = "exactly-one"
        # Grey → Green
        labels = click_term([], detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "green"
        # Green → Red (wasDetected=false → doSuppress)
        labels = click_term(labels, detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "red"
        # Red → Grey (wasDetected=false → doRestore)
        labels = click_term(labels, detected, "person", terms, mode)
        assert get_visual_state(labels, detected, "person", terms, mode) == "grey"

    def test_green_was_detected_to_blue_exclusive(self):
        """Green (was detected, exclusive) → Blue (restore)."""
        terms = ["person", "animal"]
        detected = ["person"]
        labels = ["person", "-animal"]
        new_labels = click_term(labels, detected, "person", terms, "exactly-one")
        assert get_visual_state(new_labels, detected, "person", terms, "exactly-one") == "blue"
        assert "person" not in new_labels
        assert "-person" not in new_labels

    def test_green_was_detected_to_blue_nonexclusive(self):
        """Green (was detected, non-exclusive) → Blue (restore)."""
        terms = ["photographic", "illustration"]
        detected = ["photographic"]
        labels = ["photographic"]
        new_labels = click_term(labels, detected, "photographic", terms, "zero-or-more")
        assert get_visual_state(new_labels, detected, "photographic", terms, "zero-or-more") == "blue"
        assert "photographic" not in new_labels
        assert "-photographic" not in new_labels


# ============================================================
# Category 4: Single-select - Red/suppressed term clicks
# ============================================================

class TestSingleSuppressedTerm:

    def test_red_was_grey_to_grey(self):
        """Red (was grey) → Grey (restore)."""
        terms = ["photographic"]
        detected = []
        labels = ["-photographic"]
        new_labels = click_term(labels, detected, "photographic", terms, "zero-or-more")
        assert new_labels == []
        assert get_visual_state(new_labels, detected, "photographic", terms, "zero-or-more") == "grey"

    def test_red_was_detected_to_green_any_mode(self):
        """Red (was detected) → Green (assert), regardless of mode."""
        for mode, terms in [
            ("exactly-one", ["person", "animal"]),
            ("zero-or-one", ["natural", "studio"]),
            ("zero-or-more", ["photographic", "illustration"]),
        ]:
            detected = [terms[0]]
            labels = ["-" + terms[0]]
            new_labels = click_term(labels, detected, terms[0], terms, mode)
            assert get_visual_state(new_labels, detected, terms[0], terms, mode) == "green"
            assert terms[0] in new_labels


# ============================================================
# Category 5: Single-select - Sibling effects
# ============================================================

class TestSingleSiblingEffects:

    def test_asserting_in_exclusive_removes_manual_sibling(self):
        """Asserting termB when termA is manually set removes termA."""
        terms = ["person", "animal"]
        detected = []
        labels = ["person"]
        new_labels = click_term(labels, detected, "animal", terms, "exactly-one")
        assert "animal" in new_labels
        assert "person" not in new_labels

    def test_asserting_in_exclusive_suppresses_detected_sibling(self):
        """Asserting termA suppresses detected termB."""
        terms = ["person", "animal"]
        detected = ["animal"]
        labels = []
        new_labels = click_term(labels, detected, "person", terms, "exactly-one")
        assert "person" in new_labels
        assert "-animal" in new_labels

    def test_asserting_in_exclusive_handles_mixed_siblings(self):
        """Switch from one asserted term to another with detected siblings."""
        terms = ["person", "animal", "landscape"]
        detected = ["landscape"]
        labels = ["person", "-landscape"]
        # Click animal: should remove person, keep -landscape
        new_labels = click_term(labels, detected, "animal", terms, "exactly-one")
        assert "animal" in new_labels
        assert "person" not in new_labels
        assert "-landscape" in new_labels

    def test_zero_or_more_no_sibling_suppression(self):
        """Non-exclusive: asserting one term never affects siblings."""
        terms = ["photographic", "illustration", "abstract"]
        detected = ["illustration"]
        labels = ["photographic"]
        new_labels = click_term(labels, detected, "abstract", terms, "zero-or-more")
        assert "abstract" in new_labels
        assert "photographic" in new_labels  # untouched
        assert "illustration" not in new_labels  # still just detected, not manual
        assert "-illustration" not in new_labels  # not suppressed

    def test_implicit_suppression_visual_state(self):
        """When a sibling is asserted, non-asserted terms show as red (implicitly suppressed)."""
        terms = ["person", "animal", "landscape"]
        detected = []
        labels = ["person"]
        assert get_visual_state(labels, detected, "person", terms, "exactly-one") == "green"
        assert get_visual_state(labels, detected, "animal", terms, "exactly-one") == "red"
        assert get_visual_state(labels, detected, "landscape", terms, "exactly-one") == "red"

    def test_explicit_suppression_detected_sibling(self):
        """Detected siblings show red when explicitly suppressed by exclusive assertion."""
        terms = ["person", "animal"]
        detected = ["animal"]
        labels = ["person", "-animal"]
        assert get_visual_state(labels, detected, "person", terms, "exactly-one") == "green"
        assert get_visual_state(labels, detected, "animal", terms, "exactly-one") == "red"

    def test_cross_category_independence(self):
        """Asserting in one category doesn't affect another."""
        cat1_terms = ["person", "animal"]
        cat2_terms = ["photographic", "illustration"]
        detected = ["animal", "illustration"]
        labels = []

        # Assert person in exclusive category
        labels = click_term(labels, detected, "person", cat1_terms, "exactly-one")
        assert "person" in labels
        assert "-animal" in labels

        # illustration (different category) should be unaffected
        assert "illustration" not in labels
        assert "-illustration" not in labels
        assert get_visual_state(labels, detected, "illustration", cat2_terms, "zero-or-more") == "blue"


# ============================================================
# Category 6: Multi-select (bulk) - Term cycling
# ============================================================

class TestBulkTermCycling:

    def test_bulk_grey_to_green(self, make_dataset):
        """All grey → assert green for all."""
        conn, srv, ds = make_dataset(ZERO_OR_MORE_CAT)
        try:
            imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs, "add": ["photographic"], "unsuppress": ["photographic"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "photographic" in manual
        finally:
            srv.shutdown()

    def test_bulk_green_to_red(self, make_dataset):
        """All green → suppress red for all."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            manual={"img1.jpg": ["photographic"], "img2.jpg": ["photographic"]},
            images=["img1.jpg", "img2.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs, "remove": ["photographic"], "add": ["-photographic"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "photographic" not in manual
                assert "-photographic" in manual
        finally:
            srv.shutdown()

    def test_bulk_red_nonexclusive_to_restore(self, make_dataset):
        """Red + non-exclusive → restore (remove both term and suppression)."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            manual={"img1.jpg": ["-photographic"], "img2.jpg": ["-photographic"]},
            images=["img1.jpg", "img2.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs, "remove": ["photographic"], "unsuppress": ["photographic"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "photographic" not in manual
                assert "-photographic" not in manual
        finally:
            srv.shutdown()

    def test_bulk_red_exclusive_to_green(self, make_dataset):
        """Red + exclusive → assert green (siblings removed)."""
        conn, srv, ds = make_dataset(
            EXACTLY_ONE_CAT,
            manual={"img1.jpg": ["-person"], "img2.jpg": ["-person"]},
            images=["img1.jpg", "img2.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs,
                "add": ["person"], "unsuppress": ["person"],
                "remove": ["animal", "landscape"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "person" in manual
                assert "-person" not in manual
        finally:
            srv.shutdown()

    def test_bulk_detected_to_green(self, make_dataset):
        """Detected/blue → assert green for all."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            detected={"img1.jpg": ["photographic"], "img2.jpg": ["photographic"]},
            images=["img1.jpg", "img2.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs, "add": ["photographic"], "unsuppress": ["photographic"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "photographic" in manual
        finally:
            srv.shutdown()


# ============================================================
# Category 7: Multi-select - Sibling effects
# ============================================================

class TestBulkSiblingEffects:

    def test_bulk_assert_exclusive_removes_sibling_manual(self, make_dataset):
        """Asserting termA in exclusive removes manual termB from all images."""
        conn, srv, ds = make_dataset(
            EXACTLY_ONE_CAT,
            manual={"img1.jpg": ["person"], "img2.jpg": ["animal"]},
            images=["img1.jpg", "img2.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg"]
            # Assert "landscape" for all, removing person and animal
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs,
                "add": ["landscape"], "unsuppress": ["landscape"],
                "remove": ["person", "animal"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "landscape" in manual
                assert "person" not in manual
                assert "animal" not in manual
        finally:
            srv.shutdown()

    def test_bulk_assert_exclusive_with_mixed_states(self, make_dataset):
        """Some images have termA, some have termB → assert termA for all."""
        conn, srv, ds = make_dataset(
            EXACTLY_ONE_CAT,
            manual={"img1.jpg": ["person"], "img2.jpg": ["animal"], "img3.jpg": []},
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs,
                "add": ["person"], "unsuppress": ["person"],
                "remove": ["animal", "landscape"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "person" in manual
                assert "animal" not in manual
        finally:
            srv.shutdown()


# ============================================================
# Category 8: Edge cases
# ============================================================

class TestEdgeCases:

    def test_suppression_in_put_removes_from_detected_sidecar(self, make_dataset):
        """PUT with -term removes that term from .detected.txt."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            detected={"img.jpg": ["photographic", "illustration", "abstract"]},
        )
        try:
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": ["-illustration"]})
            disk = _disk_detected(ds, "img.jpg")
            assert "illustration" not in disk
            assert "photographic" in disk
            assert "abstract" in disk
        finally:
            srv.shutdown()

    def test_multiple_suppressions_in_put(self, make_dataset):
        """Multiple suppressions in a single PUT."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            detected={"img.jpg": ["photographic", "illustration", "abstract"]},
        )
        try:
            _put_json(conn, "/api/images/img.jpg/labels", {"labels": ["-photographic", "-abstract"]})
            disk = _disk_detected(ds, "img.jpg")
            assert disk == ["illustration"]
        finally:
            srv.shutdown()

    def test_bulk_unsuppress_removes_dash_entry(self, make_dataset):
        """Bulk unsuppress removes the -term entry."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            manual={"img.jpg": ["-photographic", "other"]},
        )
        try:
            _post_json(conn, "/api/bulk/labels", {
                "images": ["img.jpg"], "unsuppress": ["photographic"],
            })
            manual, _ = _labels(conn, "img.jpg")
            assert "-photographic" not in manual
            assert "other" in manual
        finally:
            srv.shutdown()

    def test_bulk_add_idempotent(self, make_dataset):
        """Adding an already-present label does not duplicate it."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            manual={"img.jpg": ["photographic"]},
        )
        try:
            _post_json(conn, "/api/bulk/labels", {
                "images": ["img.jpg"], "add": ["photographic"],
            })
            manual, _ = _labels(conn, "img.jpg")
            assert manual.count("photographic") == 1
        finally:
            srv.shutdown()

    def test_bulk_order_unsuppress_before_remove_before_add(self, make_dataset):
        """Bulk processes: unsuppress first, then remove, then add."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT, images=["img.jpg"],
            manual={"img.jpg": ["-photographic"]},
        )
        try:
            # unsuppress removes -photographic, then add puts photographic
            _post_json(conn, "/api/bulk/labels", {
                "images": ["img.jpg"],
                "unsuppress": ["photographic"],
                "add": ["photographic"],
            })
            manual, _ = _labels(conn, "img.jpg")
            assert "photographic" in manual
            assert "-photographic" not in manual
        finally:
            srv.shutdown()

    def test_partial_state_bulk_assert_fills_all(self, make_dataset):
        """Some images have term, some don't → bulk assert gives all of them the term."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            manual={"img1.jpg": ["photographic"]},
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
        )
        try:
            imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]
            _post_json(conn, "/api/bulk/labels", {
                "images": imgs, "add": ["photographic"], "unsuppress": ["photographic"],
            })
            for name in imgs:
                manual, _ = _labels(conn, name)
                assert "photographic" in manual
        finally:
            srv.shutdown()

    def test_next_unlabeled_api(self, make_dataset):
        """The next-unlabeled endpoint finds images without .labels.txt."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            manual={"img1.jpg": ["tag"]},
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
        )
        try:
            _, data = _get_json(conn, "/api/next-unlabeled")
            assert data["name"] == "img2.jpg"
            _, data = _get_json(conn, "/api/next-unlabeled?after=img2.jpg")
            assert data["name"] == "img3.jpg"
        finally:
            srv.shutdown()

    def test_hide_labeled_filter(self, make_dataset):
        """The hide_labeled filter excludes images with .labels.txt."""
        conn, srv, ds = make_dataset(
            ZERO_OR_MORE_CAT,
            manual={"img1.jpg": ["tag"], "img2.jpg": ["tag"]},
            images=["img1.jpg", "img2.jpg", "img3.jpg"],
        )
        try:
            _, data = _get_json(conn, "/api/images?hide_labeled=1")
            names = [i["name"] for i in data["images"]]
            assert "img1.jpg" not in names
            assert "img2.jpg" not in names
            assert "img3.jpg" in names
            assert data["total"] == 1
        finally:
            srv.shutdown()
