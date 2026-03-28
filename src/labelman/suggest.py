"""Suggest workflow: propose taxonomy terms from image analysis."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path

from .integrations import run_blip, run_clip
from .schema import CategoryMode, TermList


@dataclass
class Proposal:
    """A single proposed term within a category."""
    term: str
    score: Optional[float] = None
    source_count: int = 1


@dataclass
class CategoryProposal:
    """Proposed terms for a category (expand) or a new category (bootstrap)."""
    category: str
    question: Optional[str] = None
    answers: list[str] = field(default_factory=list)
    terms: list[Proposal] = field(default_factory=list)


@dataclass
class ImageSuggestion:
    """Per-image suggestion data for sidecar writing."""
    image: str
    labels: list[str] = field(default_factory=list)
    """Detected labels from existing categories (CLIP-detected terms / VQA answers)."""
    mined_terms: list[str] = field(default_factory=list)
    """Newly mined terms extracted from captions/answers."""


@dataclass
class SuggestResult:
    """Output of a suggest run."""
    mode: str
    proposals: list[CategoryProposal] = field(default_factory=list)
    image_suggestions: list[ImageSuggestion] = field(default_factory=list)


# Common stop words filtered from bootstrap captions
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from and or but not no it its this that these those "
    "very much so too also just only".split()
)


def _extract_words(caption: str) -> list[str]:
    """Extract meaningful words from a caption, filtering stop words."""
    words = re.findall(r"[a-z]+", caption.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


def _get_answer(entry: dict) -> str:
    """Get the text from a BLIP response (caption or VQA answer).

    Handles BLIP-2's tendency to echo the prompt by stripping
    'Question: ... Answer:' prefixes from VQA responses.
    """
    text = entry.get("answer") or entry.get("caption") or ""
    # Strip echoed "Question: ... Answer:" prefix from BLIP-2 VQA
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip()


def _run_question(
    term_list: TermList,
    paths: list[str],
    question: str,
    existing_terms: set[str],
    max_tokens: int | None = None,
) -> tuple[CategoryProposal, dict[str, str]]:
    """Run a question against images and extract proposals.

    Returns:
        Tuple of (CategoryProposal, per_image_answers) where per_image_answers
        maps image path to the VQA answer for that image.
    """
    responses = run_blip(term_list, paths, prompt=question, max_tokens=max_tokens)

    # Collect raw answers
    answers: list[str] = []
    word_counts: Counter[str] = Counter()
    per_image_answers: dict[str, str] = {}

    for entry in responses:
        answer = _get_answer(entry)
        if answer:
            answers.append(answer)
            per_image_answers[entry["image"]] = answer
            words = _extract_words(answer)
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))

    min_count = 2 if len(paths) >= 4 else 1
    candidates = [
        Proposal(term=word, source_count=count)
        for word, count in word_counts.most_common()
        if count >= min_count
    ]

    # Deduplicate answers, preserve order
    seen = set()
    unique_answers = []
    for a in answers:
        key = a.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_answers.append(a.strip())

    return CategoryProposal(
        category="",  # caller sets this
        question=question,
        answers=unique_answers,
        terms=candidates,
    ), per_image_answers


def _detect_labels(term_list: TermList, flat_scores: dict[str, float]) -> list[str]:
    """Detect labels for an image from flat CLIP scores using category rules."""
    labels: list[str] = []
    for cat in term_list.categories:
        cat_terms = [
            (t.term, flat_scores.get(t.term, 0.0), term_list.effective_threshold(cat, t))
            for t in cat.terms
        ]
        if not cat_terms:
            continue

        if cat.mode == CategoryMode.EXACTLY_ONE:
            best = max(cat_terms, key=lambda c: c[1])
            labels.append(best[0])
        elif cat.mode == CategoryMode.ZERO_OR_ONE:
            above = [(name, score) for name, score, thresh in cat_terms if score >= thresh]
            if above:
                best = max(above, key=lambda c: c[1])
                labels.append(best[0])
        elif cat.mode == CategoryMode.ZERO_OR_MORE:
            labels.extend(name for name, score, thresh in cat_terms if score >= thresh)

    return labels


def bootstrap(
    term_list: TermList,
    image_paths: list[str],
    sample: Optional[int] = None,
) -> SuggestResult:
    """Bootstrap mode: generate category/term proposals from BLIP captions.

    For categories with a question, uses VQA to get targeted answers.
    For everything else, uses open-ended captioning.
    """
    paths = image_paths
    if sample is not None and sample < len(paths):
        paths = random.sample(paths, sample)

    existing_terms: set[str] = set()
    for cat in term_list.categories:
        for t in cat.terms:
            existing_terms.add(t.term.lower())

    proposals = []

    # Per-image tracking
    img_suggestions: dict[str, ImageSuggestion] = {
        p: ImageSuggestion(image=p) for p in paths
    }

    # Categories with questions get targeted VQA
    for cat in term_list.categories:
        if not cat.question:
            continue

        cp, per_image = _run_question(term_list, paths, cat.question, existing_terms,
                                      max_tokens=cat.question_answer_max_tokens)
        cp.category = cat.name
        if cp.answers or cp.terms:
            proposals.append(cp)
        # Classify per-image answers: matching taxonomy terms → labels, rest → mined
        cat_term_set = {t.term.lower() for t in cat.terms}
        for img_path, answer in per_image.items():
            if img_path not in img_suggestions:
                continue
            answer_words = set(_extract_words(answer))
            matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                       or t.term.lower() == answer.strip().lower()]
            if matched:
                img_suggestions[img_path].labels.extend(matched)
            else:
                img_suggestions[img_path].mined_terms.append(answer)

    # Open-ended captioning for general discovery
    captions = run_blip(term_list, paths)

    word_counts: Counter[str] = Counter()
    for entry in captions:
        words = _extract_words(_get_answer(entry))
        word_counts.update(set(words))  # count each word once per image
        # Track per-image mined terms
        img = entry["image"]
        if img in img_suggestions:
            new_words = [w for w in words if w not in existing_terms]
            img_suggestions[img].mined_terms.extend(new_words)

    min_count = 2 if len(paths) >= 4 else 1
    candidates = [
        Proposal(term=word, source_count=count)
        for word, count in word_counts.most_common()
        if count >= min_count
    ]

    if candidates:
        proposals.append(CategoryProposal(
            category="suggested",
            terms=candidates,
        ))

    return SuggestResult(
        mode="bootstrap",
        proposals=proposals,
        image_suggestions=list(img_suggestions.values()),
    )


def expand(
    term_list: TermList,
    image_paths: list[str],
    sample: Optional[int] = None,
) -> SuggestResult:
    """Expand mode: propose new terms within existing categories.

    Uses CLIP to score images against existing terms, then uses BLIP
    (with per-category questions when available) to caption images that
    score poorly, extracting candidate terms.
    """
    paths = image_paths
    if sample is not None and sample < len(paths):
        paths = random.sample(paths, sample)

    # Collect existing terms for filtering
    existing_terms = set()
    for cat in term_list.categories:
        for t in cat.terms:
            existing_terms.add(t.term.lower())

    # Get CLIP scores for existing terms (may be empty if no terms)
    clip_results = run_clip(term_list, paths)

    # Find images where no term scores well (potential gaps)
    threshold = term_list.defaults.threshold
    weak_images = []
    for entry in clip_results:
        scores = entry.get("scores", {})
        if not scores or max(scores.values()) < threshold:
            weak_images.append(entry["image"])

    proposals = []

    # Per-image tracking — initialize with CLIP-detected labels
    img_suggestions: dict[str, ImageSuggestion] = {}
    for entry in clip_results:
        image = entry["image"]
        flat_scores = entry.get("scores", {})
        detected = _detect_labels(term_list, flat_scores)
        img_suggestions[image] = ImageSuggestion(image=image, labels=detected)

    # For categories with questions, use targeted BLIP prompts on all images
    for cat in term_list.categories:
        if not cat.question:
            continue

        cp, per_image = _run_question(term_list, paths, cat.question, existing_terms,
                                      max_tokens=cat.question_answer_max_tokens)
        cp.category = cat.name
        if cp.answers or cp.terms:
            proposals.append(cp)
        # Classify per-image answers: matching taxonomy terms → labels, rest → mined
        for img_path, answer in per_image.items():
            if img_path not in img_suggestions:
                continue
            answer_words = set(_extract_words(answer))
            matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                       or t.term.lower() == answer.strip().lower()]
            if matched:
                img_suggestions[img_path].labels.extend(matched)
            else:
                img_suggestions[img_path].mined_terms.append(answer)

    # For weak images (no good CLIP match), use untargeted BLIP captions
    if weak_images:
        captions = run_blip(term_list, weak_images)

        word_counts: Counter[str] = Counter()
        for entry in captions:
            words = _extract_words(_get_answer(entry))
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))
            # Track per-image mined terms
            img = entry["image"]
            if img in img_suggestions:
                img_suggestions[img].mined_terms.extend(new_words)

        min_count = 2 if len(weak_images) >= 4 else 1
        candidates = [
            Proposal(term=word, source_count=count)
            for word, count in word_counts.most_common()
            if count >= min_count
        ]

        if candidates:
            proposals.append(CategoryProposal(
                category="uncategorized",
                terms=candidates,
            ))

    # Report per-category score distributions (only for categories with terms)
    for cat in term_list.categories:
        cat_terms = [t.term for t in cat.terms]
        if not cat_terms:
            continue

        top_counts: Counter[str] = Counter()
        for entry in clip_results:
            scores = entry.get("scores", {})
            cat_scores = {t: scores.get(t, 0.0) for t in cat_terms}
            if cat_scores:
                best = max(cat_scores, key=lambda t: cat_scores[t])
                top_counts[best] += 1

        unused = [t for t in cat_terms if top_counts[t] == 0]
        if unused:
            proposals.append(CategoryProposal(
                category=cat.name,
                terms=[Proposal(term=f"[unused: {t}]", source_count=0) for t in unused],
            ))

    return SuggestResult(
        mode="expand",
        proposals=proposals,
        image_suggestions=list(img_suggestions.values()),
    )


def format_suggest_result(result: SuggestResult) -> str:
    """Format a SuggestResult as human-readable YAML-like output."""
    lines = [f"# labelman suggest ({result.mode} mode)", ""]

    if not result.proposals:
        lines.append("# No proposals generated.")
        return "\n".join(lines) + "\n"

    for cp in result.proposals:
        lines.append(f"- category: {cp.category}")
        if cp.question:
            lines.append(f"  question: \"{cp.question}\"")
        if cp.answers:
            lines.append("  answers:")
            for a in cp.answers:
                lines.append(f"    - \"{a}\"")
        if cp.terms:
            lines.append("  terms:")
            for p in cp.terms:
                parts = [f"    - term: {p.term}"]
                if p.score is not None:
                    parts.append(f"  # score: {p.score:.3f}")
                if p.source_count > 0:
                    parts.append(f"  # seen in {p.source_count} image(s)")
                lines.append("".join(parts))
        lines.append("")

    return "\n".join(lines) + "\n"


def suggest_to_caption(
    term_list: TermList,
    suggestion: ImageSuggestion,
    include_mined: bool = False,
) -> str:
    """Build a comma-separated caption from an ImageSuggestion.

    Includes global_terms and detected labels by default.
    When include_mined is True, also appends mined terms.
    """
    parts = list(term_list.global_terms) + suggestion.labels
    if include_mined:
        parts.extend(suggestion.mined_terms)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return ", ".join(unique)


def write_suggest_sidecar(
    term_list: TermList,
    suggestion: ImageSuggestion,
    include_mined: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Write a .detected.txt sidecar file with suggest results for an image.

    By default writes global_terms and detected labels from existing
    categories. When include_mined is True, also writes mined terms.
    """
    image_path = Path(suggestion.image)
    if output_dir is not None:
        sidecar = output_dir / (image_path.stem + ".detected.txt")
    else:
        sidecar = image_path.with_suffix(".detected.txt")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(suggest_to_caption(term_list, suggestion, include_mined))
    return sidecar
