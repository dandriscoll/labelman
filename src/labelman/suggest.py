"""Suggest workflow: propose taxonomy terms from image analysis."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from pathlib import Path

from .integrations import run_blip, run_clip, run_llm, run_qwen_vl, run_qwen_vl_describe, run_qwen_vl_vqa
from .schema import Category, CategoryMode, TermList

logger = logging.getLogger("labelman")

# Callback type: (suggestion, current_index, total_count) -> None
_ImageCallback = Callable[["ImageSuggestion", int, int], None]


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


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate a list of strings, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


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


_MODE_INSTRUCTIONS = {
    CategoryMode.EXACTLY_ONE: (
        "Pick the single best matching label. Output only that label, nothing else."
    ),
    CategoryMode.ZERO_OR_ONE: (
        "Pick the single best matching label. "
        "If none of the labels apply, output none. "
        "Output only the label (or none), nothing else."
    ),
    CategoryMode.ZERO_OR_MORE: (
        "Pick every label that applies. "
        "If multiple labels apply, output them separated by commas. "
        "If none of the labels apply, output none. "
        "Output only labels (or none), nothing else."
    ),
}


def _classify_answer(
    term_list: TermList,
    answer: str,
    category: Category,
) -> list[str]:
    """Classify a BLIP VQA answer into category terms using the LLM.

    The system prompt varies by category mode:
      - exactly-one:  pick one label
      - zero-or-one:  pick one label or "none"
      - zero-or-more: pick all that apply, or "none"

    Returns a list of matched term strings (may be empty).
    """
    llm_config = term_list.integrations.llm
    if llm_config is None:
        return []

    term_names = [t.term for t in category.terms]
    if not term_names:
        return []

    options = ", ".join(term_names)
    instruction = _MODE_INSTRUCTIONS[category.mode]
    system_prompt = (
        f"An image was described by a vision model in response to the question: "
        f"\"{category.question}\"\n"
        f"Classify that description into one of these labels: {options}\n"
        f"{instruction}"
    )

    logger.debug("llm classify [%s, %s]: answer=%r options=[%s]",
                 category.name, category.mode.value, answer, options)
    try:
        result = run_llm(llm_config, system_prompt, f"Vision model response: {answer}")
    except Exception as exc:
        logger.warning("llm classify [%s]: %s", category.name, exc)
        return []

    # "none" means no label applies
    if result.lower().strip() == "none":
        logger.debug("llm classify [%s]: %r -> none", category.name, answer)
        return []

    # Build a lookup for case-insensitive matching
    term_lookup = {t.term.lower(): t.term for t in category.terms}

    # Parse comma-separated response (works for single-label too)
    matched: list[str] = []
    for part in result.split(","):
        key = part.strip().lower()
        if key in term_lookup:
            matched.append(term_lookup[key])

    # Enforce category mode constraints on the parsed result
    if matched and category.mode in (CategoryMode.EXACTLY_ONE, CategoryMode.ZERO_OR_ONE):
        matched = matched[:1]

    if matched:
        logger.debug("llm classify [%s]: %r -> %s", category.name, answer, matched)
    else:
        logger.debug("llm classify [%s]: %r -> no match (llm said %r)", category.name, answer, result)
    return matched


_AFFIRMATIVE = frozenset({"yes", "yeah", "yep", "true", "correct", "right"})


def _suggest_provider(term_list: TermList) -> str:
    """Determine which provider to use for suggest workflows.

    Returns 'blip_clip' if both are configured, 'qwen_vl' if qwen_vl is
    configured, or raises ValueError if neither is available.
    """
    has_blip = term_list.integrations.blip is not None
    has_clip = term_list.integrations.clip is not None
    has_qwen = term_list.integrations.qwen_vl is not None

    if has_blip or has_clip:
        return "blip_clip"
    if has_qwen:
        return "qwen_vl"
    raise ValueError(
        "No integrations configured for suggest. "
        "Configure blip+clip or qwen_vl in labelman.yaml."
    )


def _run_term_prompts(
    term_list: TermList,
    category: Category,
    paths: list[str],
    max_tokens: int | None = None,
    one_at_a_time: bool = False,
) -> dict[str, list[str]]:
    """Run per-term yes/no BLIP prompts and return matched terms per image.

    For each term that has an `ask` field, asks BLIP the question for every
    image and interprets the response as yes/no. Returns a mapping of
    image path -> list of matched term names.
    """
    prompted_terms = [t for t in category.terms if t.ask]
    if not prompted_terms:
        return {}

    # For each term, ask BLIP positive (and optionally negative) questions
    # scores[image][term] = True/False
    scores: dict[str, dict[str, bool]] = {p: {} for p in paths}

    for t in prompted_terms:
        logger.debug("term prompt [%s/%s]: %r on %d image(s)",
                     category.name, t.term, t.ask, len(paths))
        responses = run_blip(term_list, paths, prompt=t.ask, max_tokens=max_tokens,
                            one_at_a_time=one_at_a_time)
        # Collect positive answers
        pos: dict[str, bool] = {}
        for entry in responses:
            answer = _get_answer(entry).lower().strip().rstrip(".")
            is_yes = answer in _AFFIRMATIVE or answer.startswith("yes")
            pos[entry["image"]] = is_yes
            logger.debug("term prompt [%s/%s]: %s -> %r (%s)",
                         category.name, t.term, Path(entry["image"]).name,
                         answer, "yes" if is_yes else "no")

        # If ask_negative is set, run the negative probe
        neg: dict[str, bool] = {}
        if t.ask_negative:
            logger.debug("term prompt neg [%s/%s]: %r on %d image(s)",
                         category.name, t.term, t.ask_negative, len(paths))
            neg_responses = run_blip(term_list, paths, prompt=t.ask_negative,
                                     max_tokens=max_tokens, one_at_a_time=one_at_a_time)
            for entry in neg_responses:
                answer = _get_answer(entry).lower().strip().rstrip(".")
                is_yes = answer in _AFFIRMATIVE or answer.startswith("yes")
                neg[entry["image"]] = is_yes
                logger.debug("term prompt neg [%s/%s]: %s -> %r (%s)",
                             category.name, t.term, Path(entry["image"]).name,
                             answer, "yes" if is_yes else "no")

        # Term matches when positive=yes AND (no negative probe OR negative=no)
        for img_path in paths:
            pos_yes = pos.get(img_path, False)
            neg_yes = neg.get(img_path, False)
            if t.ask_negative:
                match = pos_yes and not neg_yes
            else:
                match = pos_yes
            scores[img_path][t.term] = match

    # Apply category mode to select terms
    result: dict[str, list[str]] = {}
    for img_path in paths:
        img_scores = scores.get(img_path, {})
        affirmed = [term for term, is_yes in img_scores.items() if is_yes]

        if category.mode == CategoryMode.EXACTLY_ONE:
            # Pick one — prefer affirmed, fall back to first prompted term
            if len(affirmed) == 1:
                result[img_path] = affirmed
            elif len(affirmed) > 1:
                result[img_path] = [affirmed[0]]  # first match
            else:
                result[img_path] = []
        elif category.mode == CategoryMode.ZERO_OR_ONE:
            if len(affirmed) == 1:
                result[img_path] = affirmed
            elif len(affirmed) > 1:
                result[img_path] = [affirmed[0]]
            else:
                result[img_path] = []
        elif category.mode == CategoryMode.ZERO_OR_MORE:
            result[img_path] = affirmed

    return result


def _run_question(
    term_list: TermList,
    paths: list[str],
    question: str,
    existing_terms: set[str],
    max_tokens: int | None = None,
    one_at_a_time: bool = False,
) -> tuple[CategoryProposal, dict[str, str]]:
    """Run a question against images and extract proposals.

    Returns:
        Tuple of (CategoryProposal, per_image_answers) where per_image_answers
        maps image path to the VQA answer for that image.
    """
    responses = run_blip(term_list, paths, prompt=question, max_tokens=max_tokens,
                         one_at_a_time=one_at_a_time)

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


def _run_qwen_vl_term_prompts(
    term_list: TermList,
    category: Category,
    paths: list[str],
    max_tokens: int | None = None,
) -> dict[str, list[str]]:
    """Run per-term yes/no VQA using Qwen2.5-VL instead of BLIP.

    For each term with an `ask` field, asks qwen_vl the question and
    interprets the answer. Returns image path -> matched term names.
    """
    config = term_list.integrations.qwen_vl
    if config is None:
        return {}

    prompted_terms = [t for t in category.terms if t.ask]
    if not prompted_terms:
        return {}

    scores: dict[str, dict[str, bool]] = {p: {} for p in paths}

    for t in prompted_terms:
        logger.debug("qwen_vl term prompt [%s/%s]: %r on %d image(s)",
                     category.name, t.term, t.ask, len(paths))
        for img_path in paths:
            answer = run_qwen_vl_vqa(config, img_path, t.ask,
                                     max_tokens=max_tokens or 20)
            is_yes = answer.lower().strip().rstrip(".") in _AFFIRMATIVE or answer.lower().startswith("yes")
            scores[img_path][t.term] = is_yes
            logger.debug("qwen_vl term prompt [%s/%s]: %s -> %r (%s)",
                         category.name, t.term, Path(img_path).name,
                         answer, "yes" if is_yes else "no")

            if t.ask_negative and is_yes:
                neg_answer = run_qwen_vl_vqa(config, img_path, t.ask_negative,
                                             max_tokens=max_tokens or 20)
                neg_yes = neg_answer.lower().strip().rstrip(".") in _AFFIRMATIVE or neg_answer.lower().startswith("yes")
                if neg_yes:
                    scores[img_path][t.term] = False
                    logger.debug("qwen_vl term prompt neg [%s/%s]: %s -> %r (negated)",
                                 category.name, t.term, Path(img_path).name, neg_answer)

    # Apply category mode
    result: dict[str, list[str]] = {}
    for img_path in paths:
        affirmed = [term for term, is_yes in scores.get(img_path, {}).items() if is_yes]
        if category.mode == CategoryMode.ZERO_OR_MORE:
            result[img_path] = affirmed
        elif affirmed:
            result[img_path] = [affirmed[0]]
        else:
            result[img_path] = []
    return result


def _run_qwen_vl_question(
    term_list: TermList,
    paths: list[str],
    question: str,
    existing_terms: set[str],
    max_tokens: int | None = None,
) -> tuple[CategoryProposal, dict[str, str]]:
    """Run a VQA question against images using Qwen2.5-VL instead of BLIP.

    Returns (CategoryProposal, per_image_answers).
    """
    config = term_list.integrations.qwen_vl
    assert config is not None

    answers: list[str] = []
    word_counts: Counter[str] = Counter()
    per_image_answers: dict[str, str] = {}

    for img_path in paths:
        answer = run_qwen_vl_vqa(config, img_path, question,
                                 max_tokens=max_tokens or 100)
        if answer:
            answers.append(answer)
            per_image_answers[img_path] = answer
            words = _extract_words(answer)
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))

    min_count = 2 if len(paths) >= 4 else 1
    candidates = [
        Proposal(term=word, source_count=count)
        for word, count in word_counts.most_common()
        if count >= min_count
    ]

    unique_answers = _dedupe(answers)

    return CategoryProposal(
        category="",
        question=question,
        answers=unique_answers,
        terms=candidates,
    ), per_image_answers


def _enforce_category_modes(term_list: TermList, labels: list[str]) -> list[str]:
    """Enforce category mode constraints on a flat label list.

    For exactly-one and zero-or-one categories, keeps only the first
    matching term. For zero-or-more, keeps all. Labels not belonging to
    any category are passed through unchanged.
    """
    # Build term -> category lookup
    term_to_cat: dict[str, str] = {}
    for cat in term_list.categories:
        for t in cat.terms:
            term_to_cat[t.term] = cat.name

    # Track which categories have already emitted a label
    cat_emitted: dict[str, int] = {}
    exclusive_modes = {CategoryMode.EXACTLY_ONE, CategoryMode.ZERO_OR_ONE}
    cat_mode: dict[str, CategoryMode] = {cat.name: cat.mode for cat in term_list.categories}

    result: list[str] = []
    for label in labels:
        cat_name = term_to_cat.get(label)
        if cat_name is None:
            # Not a taxonomy term — pass through
            result.append(label)
            continue
        mode = cat_mode[cat_name]
        if mode in exclusive_modes:
            if cat_emitted.get(cat_name, 0) >= 1:
                continue  # already have one for this category
        cat_emitted[cat_name] = cat_emitted.get(cat_name, 0) + 1
        result.append(label)

    return result


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


def _bootstrap_blip_clip(
    term_list: TermList,
    paths: list[str],
    existing_terms: set[str],
    one_at_a_time: bool = False,
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Bootstrap using BLIP captions and CLIP scoring (original path)."""
    proposals = []
    img_suggestions: dict[str, ImageSuggestion] = {
        p: ImageSuggestion(image=p) for p in paths
    }

    # Categories with questions/per-term prompts get targeted VQA
    for cat in term_list.categories:
        has_term_prompts = any(t.ask for t in cat.terms)
        if not cat.question and not has_term_prompts:
            continue

        # Per-term yes/no prompts take priority
        if has_term_prompts:
            logger.debug("bootstrap: per-term prompts for category %r", cat.name)
            term_matches = _run_term_prompts(term_list, cat, paths,
                                             max_tokens=cat.question_answer_max_tokens,
                                             one_at_a_time=one_at_a_time)
            classified_answers: list[str] = []
            for img_path in paths:
                matched = term_matches.get(img_path, [])
                if matched:
                    img_suggestions[img_path].labels.extend(matched)
                    classified_answers.extend(matched)
            cp = CategoryProposal(
                category=cat.name,
                question=cat.question,
                answers=_dedupe(classified_answers),
            )
            if cp.answers:
                proposals.append(cp)
            continue

        blip_prompt = cat.ask or cat.question
        logger.debug("bootstrap: VQA for category %r: blip=%r question=%r",
                     cat.name, blip_prompt, cat.question)
        cp, per_image = _run_question(term_list, paths, blip_prompt, existing_terms,
                                      max_tokens=cat.question_answer_max_tokens,
                                      one_at_a_time=one_at_a_time)
        cp.category = cat.name
        cp.question = cat.question
        classified_answers = []
        for img_path, answer in per_image.items():
            if img_path not in img_suggestions:
                continue
            answer_words = set(_extract_words(answer))
            matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                       or t.term.lower() == answer.strip().lower()]
            if matched:
                logger.debug("bootstrap: %s [%s] word-matched %r -> %s",
                             Path(img_path).name, cat.name, answer, matched)
            if not matched:
                matched = _classify_answer(term_list, answer, cat)
            if matched:
                img_suggestions[img_path].labels.extend(matched)
                classified_answers.extend(matched)
            else:
                logger.debug("bootstrap: %s [%s] unmatched %r -> mined",
                             Path(img_path).name, cat.name, answer)
                img_suggestions[img_path].mined_terms.append(answer)
                classified_answers.append(answer)
        cp.answers = _dedupe(classified_answers)
        if cp.answers or cp.terms:
            proposals.append(cp)

    # Open-ended captioning for general discovery
    logger.debug("bootstrap: open-ended captioning on %d image(s)", len(paths))
    captions = run_blip(term_list, paths, one_at_a_time=one_at_a_time)

    word_counts: Counter[str] = Counter()
    for entry in captions:
        words = _extract_words(_get_answer(entry))
        word_counts.update(set(words))
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

    suggestions = list(img_suggestions.values())
    for sugg in suggestions:
        sugg.labels = _enforce_category_modes(term_list, sugg.labels)
    if on_image:
        for i, sugg in enumerate(suggestions, 1):
            on_image(sugg, i, len(suggestions))

    return SuggestResult(
        mode="bootstrap",
        proposals=proposals,
        image_suggestions=suggestions,
    )


def _bootstrap_qwen_vl(
    term_list: TermList,
    paths: list[str],
    existing_terms: set[str],
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Bootstrap using Qwen2.5-VL for classification and captioning.

    Processes each image through all phases (classify, VQA, describe) before
    moving to the next, so sidecars can be written incrementally via on_image.
    """
    config = term_list.integrations.qwen_vl
    assert config is not None

    # Pre-compute which categories need VQA
    vqa_cats: list[Category] = []
    prompt_cats: list[Category] = []
    for cat in term_list.categories:
        has_term_prompts = any(t.ask for t in cat.terms)
        if has_term_prompts:
            prompt_cats.append(cat)
        elif cat.question:
            vqa_cats.append(cat)

    # Per-category answer tracking (for proposals built after all images)
    cat_answers: dict[str, list[str]] = {cat.name: [] for cat in vqa_cats + prompt_cats}

    img_suggestions: list[ImageSuggestion] = []
    word_counts: Counter[str] = Counter()

    logger.debug("bootstrap: qwen_vl processing %d image(s) (classify + VQA + describe)",
                 len(paths))

    for i, img_path in enumerate(paths, 1):
        sugg = ImageSuggestion(image=img_path)

        # Phase 1: Classification against existing terms
        if any(c.terms for c in term_list.categories):
            labels = run_qwen_vl(term_list, img_path)
            for cat_name, selected in labels.items():
                if selected:
                    sugg.labels.extend(selected)

        # Phase 2: Per-term yes/no prompts
        for cat in prompt_cats:
            prompted_terms = [t for t in cat.terms if t.ask]
            for t in prompted_terms:
                answer = run_qwen_vl_vqa(config, img_path, t.ask,
                                         max_tokens=cat.question_answer_max_tokens or 20)
                is_yes = answer.lower().strip().rstrip(".") in _AFFIRMATIVE or answer.lower().startswith("yes")
                if is_yes and t.ask_negative:
                    neg = run_qwen_vl_vqa(config, img_path, t.ask_negative,
                                          max_tokens=cat.question_answer_max_tokens or 20)
                    if neg.lower().strip().rstrip(".") in _AFFIRMATIVE or neg.lower().startswith("yes"):
                        is_yes = False
                if is_yes:
                    sugg.labels.append(t.term)
                    cat_answers[cat.name].append(t.term)

        # Phase 3: Category VQA questions
        for cat in vqa_cats:
            question = cat.ask or cat.question
            answer = run_qwen_vl_vqa(config, img_path, question,
                                     max_tokens=cat.question_answer_max_tokens or 100)
            if answer:
                answer_words = set(_extract_words(answer))
                matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                           or t.term.lower() == answer.strip().lower()]
                if matched:
                    sugg.labels.extend(matched)
                    cat_answers[cat.name].extend(matched)
                else:
                    sugg.mined_terms.append(answer)
                    cat_answers[cat.name].append(answer)

        # Phase 4: Open-ended description
        description = run_qwen_vl_describe(config, img_path)
        words = _extract_words(description)
        word_counts.update(set(words))
        new_words = [w for w in words if w not in existing_terms]
        sugg.mined_terms.extend(new_words)

        # Enforce category constraints (e.g. exactly-one gets only one label)
        sugg.labels = _enforce_category_modes(term_list, sugg.labels)

        img_suggestions.append(sugg)
        if on_image:
            on_image(sugg, i, len(paths))

    # Build proposals from accumulated answers
    proposals = []
    for cat in prompt_cats + vqa_cats:
        answers = cat_answers.get(cat.name, [])
        if answers:
            proposals.append(CategoryProposal(
                category=cat.name,
                question=cat.question,
                answers=_dedupe(answers),
            ))

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
        image_suggestions=img_suggestions,
    )


def bootstrap(
    term_list: TermList,
    image_paths: list[str],
    sample: Optional[int] = None,
    one_at_a_time: bool = False,
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Bootstrap mode: generate category/term proposals from image analysis.

    Uses BLIP/CLIP when configured, or Qwen2.5-VL as a standalone alternative.
    When on_image is provided, it is called per-image as each suggestion is
    ready, allowing incremental sidecar writes.
    """
    paths = image_paths
    if sample is not None and sample < len(paths):
        paths = random.sample(paths, sample)

    logger.debug("bootstrap: %d image(s)%s", len(paths),
                 f" (sampled from {len(image_paths)})" if sample else "")

    existing_terms: set[str] = set()
    for cat in term_list.categories:
        for t in cat.terms:
            existing_terms.add(t.term.lower())

    provider = _suggest_provider(term_list)
    logger.debug("bootstrap: using provider %s", provider)

    if provider == "qwen_vl":
        return _bootstrap_qwen_vl(term_list, paths, existing_terms, on_image=on_image)
    return _bootstrap_blip_clip(term_list, paths, existing_terms,
                                one_at_a_time=one_at_a_time, on_image=on_image)


def _expand_blip_clip(
    term_list: TermList,
    paths: list[str],
    existing_terms: set[str],
    one_at_a_time: bool = False,
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Expand using CLIP scoring and BLIP captioning (original path)."""
    # Get CLIP scores for existing terms (may be empty if no terms)
    clip_results = run_clip(term_list, paths, one_at_a_time=one_at_a_time)

    # Find images where no term scores well (potential gaps)
    threshold = term_list.defaults.threshold
    weak_images = []
    for entry in clip_results:
        scores = entry.get("scores", {})
        if not scores or max(scores.values()) < threshold:
            weak_images.append(entry["image"])

    logger.debug("expand: %d weak image(s) (below threshold %.2f)", len(weak_images), threshold)

    proposals = []

    # Per-image tracking — initialize with CLIP-detected labels
    img_suggestions: dict[str, ImageSuggestion] = {}
    for entry in clip_results:
        image = entry["image"]
        flat_scores = entry.get("scores", {})
        detected = _detect_labels(term_list, flat_scores)
        img_suggestions[image] = ImageSuggestion(image=image, labels=detected)
        if detected:
            logger.debug("expand: %s CLIP detected: %s", Path(image).name, detected)

    # For categories with questions/per-term prompts, use targeted BLIP prompts
    for cat in term_list.categories:
        has_term_prompts = any(t.ask for t in cat.terms)
        if not cat.question and not has_term_prompts:
            continue

        if has_term_prompts:
            logger.debug("expand: per-term prompts for category %r", cat.name)
            term_matches = _run_term_prompts(term_list, cat, paths,
                                             max_tokens=cat.question_answer_max_tokens,
                                             one_at_a_time=one_at_a_time)
            classified_answers: list[str] = []
            for img_path in paths:
                matched = term_matches.get(img_path, [])
                if matched:
                    if img_path in img_suggestions:
                        img_suggestions[img_path].labels.extend(matched)
                    classified_answers.extend(matched)
            cp = CategoryProposal(
                category=cat.name,
                question=cat.question,
                answers=_dedupe(classified_answers),
            )
            if cp.answers:
                proposals.append(cp)
            continue

        blip_prompt = cat.ask or cat.question
        logger.debug("expand: VQA for category %r: blip=%r question=%r",
                     cat.name, blip_prompt, cat.question)
        cp, per_image = _run_question(term_list, paths, blip_prompt, existing_terms,
                                      max_tokens=cat.question_answer_max_tokens,
                                      one_at_a_time=one_at_a_time)
        cp.category = cat.name
        cp.question = cat.question
        classified_answers = []
        for img_path, answer in per_image.items():
            if img_path not in img_suggestions:
                continue
            answer_words = set(_extract_words(answer))
            matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                       or t.term.lower() == answer.strip().lower()]
            if matched:
                logger.debug("expand: %s [%s] word-matched %r -> %s",
                             Path(img_path).name, cat.name, answer, matched)
            if not matched:
                matched = _classify_answer(term_list, answer, cat)
            if matched:
                img_suggestions[img_path].labels.extend(matched)
                classified_answers.extend(matched)
            else:
                logger.debug("expand: %s [%s] unmatched %r -> mined",
                             Path(img_path).name, cat.name, answer)
                img_suggestions[img_path].mined_terms.append(answer)
                classified_answers.append(answer)
        cp.answers = _dedupe(classified_answers)
        if cp.answers or cp.terms:
            proposals.append(cp)

    # For weak images (no good CLIP match), use untargeted BLIP captions
    if weak_images:
        logger.debug("expand: captioning %d weak image(s)", len(weak_images))
        captions = run_blip(term_list, weak_images, one_at_a_time=one_at_a_time)

        word_counts: Counter[str] = Counter()
        for entry in captions:
            words = _extract_words(_get_answer(entry))
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))
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

    suggestions = list(img_suggestions.values())
    for sugg in suggestions:
        sugg.labels = _enforce_category_modes(term_list, sugg.labels)
    if on_image:
        for i, sugg in enumerate(suggestions, 1):
            on_image(sugg, i, len(suggestions))

    return SuggestResult(
        mode="expand",
        proposals=proposals,
        image_suggestions=suggestions,
    )


def _expand_qwen_vl(
    term_list: TermList,
    paths: list[str],
    existing_terms: set[str],
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Expand using Qwen2.5-VL for classification and captioning.

    Processes each image through all phases (classify, VQA, describe if weak)
    before moving to the next, so sidecars can be written incrementally.
    """
    config = term_list.integrations.qwen_vl
    assert config is not None

    # Pre-compute which categories need VQA
    vqa_cats: list[Category] = []
    prompt_cats: list[Category] = []
    for cat in term_list.categories:
        has_term_prompts = any(t.ask for t in cat.terms)
        if has_term_prompts:
            prompt_cats.append(cat)
        elif cat.question:
            vqa_cats.append(cat)

    cat_answers: dict[str, list[str]] = {cat.name: [] for cat in vqa_cats + prompt_cats}
    img_suggestions: list[ImageSuggestion] = []
    word_counts: Counter[str] = Counter()

    logger.debug("expand: qwen_vl processing %d image(s)", len(paths))

    for i, img_path in enumerate(paths, 1):
        sugg = ImageSuggestion(image=img_path)
        is_weak = True

        # Phase 1: Classification
        if any(c.terms for c in term_list.categories):
            labels = run_qwen_vl(term_list, img_path)
            flat_labels = [t for terms in labels.values() for t in terms]
            sugg.labels.extend(flat_labels)
            if flat_labels:
                is_weak = False
                logger.debug("expand: %s qwen_vl detected: %s", Path(img_path).name, flat_labels)

        # Phase 2: Per-term yes/no prompts
        for cat in prompt_cats:
            prompted_terms = [t for t in cat.terms if t.ask]
            for t in prompted_terms:
                answer = run_qwen_vl_vqa(config, img_path, t.ask,
                                         max_tokens=cat.question_answer_max_tokens or 20)
                is_yes = answer.lower().strip().rstrip(".") in _AFFIRMATIVE or answer.lower().startswith("yes")
                if is_yes and t.ask_negative:
                    neg = run_qwen_vl_vqa(config, img_path, t.ask_negative,
                                          max_tokens=cat.question_answer_max_tokens or 20)
                    if neg.lower().strip().rstrip(".") in _AFFIRMATIVE or neg.lower().startswith("yes"):
                        is_yes = False
                if is_yes:
                    sugg.labels.append(t.term)
                    cat_answers[cat.name].append(t.term)

        # Phase 3: Category VQA questions
        for cat in vqa_cats:
            question = cat.ask or cat.question
            answer = run_qwen_vl_vqa(config, img_path, question,
                                     max_tokens=cat.question_answer_max_tokens or 100)
            if answer:
                answer_words = set(_extract_words(answer))
                matched = [t.term for t in cat.terms if t.term.lower() in answer_words
                           or t.term.lower() == answer.strip().lower()]
                if matched:
                    sugg.labels.extend(matched)
                    cat_answers[cat.name].extend(matched)
                else:
                    sugg.mined_terms.append(answer)
                    cat_answers[cat.name].append(answer)

        # Phase 4: Describe weak images for term mining
        if is_weak:
            description = run_qwen_vl_describe(config, img_path)
            words = _extract_words(description)
            new_words = [w for w in words if w not in existing_terms]
            word_counts.update(set(new_words))
            sugg.mined_terms.extend(new_words)

        # Enforce category constraints (e.g. exactly-one gets only one label)
        sugg.labels = _enforce_category_modes(term_list, sugg.labels)

        img_suggestions.append(sugg)
        if on_image:
            on_image(sugg, i, len(paths))

    # Build proposals
    proposals = []
    for cat in prompt_cats + vqa_cats:
        answers = cat_answers.get(cat.name, [])
        if answers:
            proposals.append(CategoryProposal(
                category=cat.name,
                question=cat.question,
                answers=_dedupe(answers),
            ))

    # Uncategorized term proposals from weak image descriptions
    if word_counts:
        weak_count = sum(1 for s in img_suggestions if not s.labels)
        min_count = 2 if weak_count >= 4 else 1
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

    # Report unused terms
    term_counts: Counter[str] = Counter()
    for sugg in img_suggestions:
        term_counts.update(sugg.labels)
    for cat in term_list.categories:
        cat_terms = [t.term for t in cat.terms]
        if not cat_terms:
            continue
        unused = [t for t in cat_terms if term_counts[t] == 0]
        if unused:
            proposals.append(CategoryProposal(
                category=cat.name,
                terms=[Proposal(term=f"[unused: {t}]", source_count=0) for t in unused],
            ))

    return SuggestResult(
        mode="expand",
        proposals=proposals,
        image_suggestions=img_suggestions,
    )


def expand(
    term_list: TermList,
    image_paths: list[str],
    sample: Optional[int] = None,
    one_at_a_time: bool = False,
    on_image: Optional[_ImageCallback] = None,
) -> SuggestResult:
    """Expand mode: propose new terms within existing categories.

    Uses BLIP/CLIP when configured, or Qwen2.5-VL as a standalone alternative.
    When on_image is provided, it is called per-image as each suggestion is
    ready, allowing incremental sidecar writes.
    """
    paths = image_paths
    if sample is not None and sample < len(paths):
        paths = random.sample(paths, sample)

    logger.debug("expand: %d image(s)%s", len(paths),
                 f" (sampled from {len(image_paths)})" if sample else "")

    existing_terms = set()
    for cat in term_list.categories:
        for t in cat.terms:
            existing_terms.add(t.term.lower())

    provider = _suggest_provider(term_list)
    logger.debug("expand: using provider %s", provider)

    if provider == "qwen_vl":
        return _expand_qwen_vl(term_list, paths, existing_terms, on_image=on_image)
    return _expand_blip_clip(term_list, paths, existing_terms,
                             one_at_a_time=one_at_a_time, on_image=on_image)


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
