"""Suggest workflow: propose taxonomy terms from image analysis."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .integrations import run_blip, run_clip
from .schema import TermList


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
class SuggestResult:
    """Output of a suggest run."""
    mode: str
    proposals: list[CategoryProposal] = field(default_factory=list)


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
) -> CategoryProposal:
    """Run a question against images and extract proposals."""
    responses = run_blip(term_list, paths, prompt=question, max_tokens=max_tokens)

    # Collect raw answers
    answers: list[str] = []
    word_counts: Counter[str] = Counter()

    for entry in responses:
        answer = _get_answer(entry)
        if answer:
            answers.append(answer)
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
    )


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
        paths = paths[:sample]

    existing_terms: set[str] = set()
    for cat in term_list.categories:
        for t in cat.terms:
            existing_terms.add(t.term.lower())

    proposals = []

    # Categories with questions get targeted VQA
    for cat in term_list.categories:
        if not cat.question:
            continue

        cp = _run_question(term_list, paths, cat.question, existing_terms,
                           max_tokens=cat.question_answer_max_tokens)
        cp.category = cat.name
        if cp.answers or cp.terms:
            proposals.append(cp)

    # Open-ended captioning for general discovery
    captions = run_blip(term_list, paths)

    word_counts: Counter[str] = Counter()
    for entry in captions:
        words = _extract_words(_get_answer(entry))
        word_counts.update(set(words))  # count each word once per image

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

    return SuggestResult(mode="bootstrap", proposals=proposals)


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
        paths = paths[:sample]

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

    # For categories with questions, use targeted BLIP prompts on all images
    for cat in term_list.categories:
        if not cat.question:
            continue

        cp = _run_question(term_list, paths, cat.question, existing_terms,
                           max_tokens=cat.question_answer_max_tokens)
        cp.category = cat.name
        if cp.answers or cp.terms:
            proposals.append(cp)

    # For weak images (no good CLIP match), use untargeted BLIP captions
    if weak_images:
        captions = run_blip(term_list, weak_images)

        word_counts: Counter[str] = Counter()
        for entry in captions:
            words = _extract_words(_get_answer(entry))
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))

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

    return SuggestResult(mode="expand", proposals=proposals)


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
