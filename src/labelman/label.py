"""Labeling engine: apply taxonomy to scored images, produce outputs."""

from __future__ import annotations

import csv
import html as html_mod
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

from .schema import CategoryMode, TermList


@dataclass
class ImageLabels:
    image: str
    labels: dict[str, list[str]]
    """Mapping of category name to list of assigned term strings."""
    all_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    """All raw scores for every term, including suppressed ones."""
    final_labels: list[str] = field(default_factory=list)
    """Deduplicated merged labels: global + manual + detected, in order."""


# Scores type: {category_name: {term_name: float}}
Scores = dict[str, dict[str, float]]


def apply_labels(term_list: TermList, image: str, scores: Scores) -> ImageLabels:
    """Apply the term list to a single image given its scores.

    Returns ImageLabels with assigned labels per category and all raw scores.
    """
    labels: dict[str, list[str]] = {}

    for category in term_list.categories:
        cat_scores = scores.get(category.name, {})

        candidates: list[tuple[str, float, float]] = []
        for term in category.terms:
            score = cat_scores.get(term.term, 0.0)
            threshold = term_list.effective_threshold(category, term)
            candidates.append((term.term, score, threshold))

        if category.mode == CategoryMode.EXACTLY_ONE:
            best = max(candidates, key=lambda c: c[1])
            labels[category.name] = [best[0]]

        elif category.mode == CategoryMode.ZERO_OR_ONE:
            above = [(name, score) for name, score, thresh in candidates if score >= thresh]
            if above:
                best = max(above, key=lambda c: c[1])
                labels[category.name] = [best[0]]
            else:
                labels[category.name] = []

        elif category.mode == CategoryMode.ZERO_OR_MORE:
            labels[category.name] = [
                name for name, score, thresh in candidates if score >= thresh
            ]

    return ImageLabels(image=image, labels=labels, all_scores=scores)


def load_manual_sidecar(image_path: str) -> list[str]:
    """Load manual labels from a .labels.txt sidecar file if it exists.

    Sidecar naming: for image_001.jpg, the sidecar is image_001.labels.txt.
    Format: comma-separated labels, whitespace-trimmed, empty entries ignored.
    Returns an empty list if no sidecar exists.
    """
    p = Path(image_path)
    sidecar = p.with_suffix(".labels.txt")
    if not sidecar.is_file():
        return []
    text = sidecar.read_text().strip()
    if not text:
        return []
    return [label.strip() for label in text.split(",") if label.strip()]


def assemble_final_labels(
    term_list: TermList,
    image: str,
    scores: Scores,
    manual_labels: list[str] | None = None,
) -> ImageLabels:
    """Assemble final labels for an image from all sources.

    Merge order:
      1. global_terms from labelman.yaml (applied to every image)
      2. manual sidecar labels (per-image overrides)
      3. detected labels (from category/threshold rules)

    Global and manual labels bypass detection, thresholds, and category
    semantics. They are literal labels. The final list is deduplicated
    while preserving first-occurrence order.
    """
    detected = apply_labels(term_list, image, scores)

    # Collect detected labels in category order
    detected_flat: list[str] = []
    for cat_terms in detected.labels.values():
        detected_flat.extend(cat_terms)

    if manual_labels is None:
        manual_labels = []

    # Separate suppression directives (-term) from additive manual labels
    suppressions: set[str] = set()
    additive_manual: list[str] = []
    for label in manual_labels:
        if label.startswith("-"):
            suppressions.add(label[1:])
        else:
            additive_manual.append(label)

    # Implicit suppression: in exclusive categories (exactly-one, zero-or-one),
    # manually selecting a term displaces all other terms in that category.
    additive_set = set(additive_manual)
    for cat in term_list.categories:
        if cat.mode in (CategoryMode.EXACTLY_ONE, CategoryMode.ZERO_OR_ONE):
            cat_terms = [t.term for t in cat.terms]
            manual_in_cat = [t for t in cat_terms if t in additive_set]
            if manual_in_cat:
                for t in cat_terms:
                    if t not in additive_set:
                        suppressions.add(t)

    # Merge in order: global, manual, detected — deduplicate and suppress
    merged: list[str] = []
    seen: set[str] = set()
    for label in [*term_list.global_terms, *additive_manual, *detected_flat]:
        if label not in seen and label not in suppressions:
            merged.append(label)
            seen.add(label)

    result = ImageLabels(
        image=image,
        labels=detected.labels,
        all_scores=scores,
        final_labels=merged,
    )
    return result


def labels_to_caption(result: ImageLabels) -> str:
    """Convert image labels to a stable-diffusion-style comma-separated caption.

    Uses final_labels if available (includes global/manual/detected).
    Falls back to category labels only.
    """
    if result.final_labels:
        return ", ".join(result.final_labels)
    terms = []
    for cat_terms in result.labels.values():
        terms.extend(cat_terms)
    return ", ".join(terms)


def detected_to_caption(result: ImageLabels) -> str:
    """Convert detected category labels to a comma-separated caption.

    Only includes labels from category detection (no global terms, no manual labels).
    """
    terms = []
    for cat_terms in result.labels.values():
        terms.extend(cat_terms)
    return ", ".join(terms)


def write_sidecar(result: ImageLabels, output_dir: Path | None = None) -> Path:
    """Write a .detected.txt sidecar file with detected category labels for an image.

    Only writes labels from category detection (CLIP scores).
    Global terms and manual labels are merged at apply time.
    """
    image_path = Path(result.image)
    if output_dir is not None:
        sidecar = output_dir / (image_path.stem + ".detected.txt")
    else:
        sidecar = image_path.with_suffix(".detected.txt")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(detected_to_caption(result))
    return sidecar


def load_detected_sidecar(image_path: str) -> list[str]:
    """Load detected labels from a .detected.txt sidecar file if it exists.

    Format: comma-separated labels, whitespace-trimmed, empty entries ignored.
    Returns an empty list if no sidecar exists.
    """
    p = Path(image_path)
    sidecar = p.with_suffix(".detected.txt")
    if not sidecar.is_file():
        return []
    text = sidecar.read_text().strip()
    if not text:
        return []
    return [label.strip() for label in text.split(",") if label.strip()]


def merge_sidecars(
    term_list: TermList,
    image_path: str,
) -> list[str]:
    """Merge manual (.labels.txt) and detected (.detected.txt) into final labels.

    Applies the same merge logic as assemble_final_labels:
      1. global_terms (from labelman.yaml)
      2. additive manual labels
      3. detected labels
    Minus explicit suppressions (-term) and implicit exclusive-category suppressions.
    """
    manual_labels = load_manual_sidecar(image_path)
    detected_labels = load_detected_sidecar(image_path)

    # Separate suppression directives from additive manual labels
    suppressions: set[str] = set()
    additive_manual: list[str] = []
    for label in manual_labels:
        if label.startswith("-"):
            suppressions.add(label[1:])
        else:
            additive_manual.append(label)

    # Implicit suppression for exclusive categories
    additive_set = set(additive_manual)
    for cat in term_list.categories:
        if cat.mode in (CategoryMode.EXACTLY_ONE, CategoryMode.ZERO_OR_ONE):
            cat_terms = [t.term for t in cat.terms]
            manual_in_cat = [t for t in cat_terms if t in additive_set]
            if manual_in_cat:
                for t in cat_terms:
                    if t not in additive_set:
                        suppressions.add(t)

    # Merge in order: global, manual, detected — deduplicate and suppress
    merged: list[str] = []
    seen: set[str] = set()
    for label in [*term_list.global_terms, *additive_manual, *detected_labels]:
        if label not in seen and label not in suppressions:
            merged.append(label)
            seen.add(label)

    return merged


def write_final_sidecar(image_path: str, labels: list[str], output_dir: Path | None = None) -> Path:
    """Write the final merged .txt sidecar for an image."""
    p = Path(image_path)
    if output_dir is not None:
        sidecar = output_dir / (p.stem + ".txt")
    else:
        sidecar = p.with_suffix(".txt")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(", ".join(labels))
    return sidecar


def write_csv(
    term_list: TermList,
    results: list[ImageLabels],
    output_path: Path,
) -> None:
    """Write a CSV with all scores for all images, including suppressed ones.

    Columns: image, then for each category/term pair: category/term (score),
    plus an 'assigned' column indicating whether it was selected.
    """
    # Build column order from taxonomy
    col_specs: list[tuple[str, str]] = []  # (category, term)
    for cat in term_list.categories:
        for t in cat.terms:
            col_specs.append((cat.name, t.term))

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["image"]
        for cat_name, term_name in col_specs:
            header.append(f"{cat_name}/{term_name}_score")
            header.append(f"{cat_name}/{term_name}_assigned")
        header.append("caption")
        writer.writerow(header)

        # Rows
        for result in results:
            row: list[str] = [result.image]
            for cat_name, term_name in col_specs:
                score = result.all_scores.get(cat_name, {}).get(term_name, 0.0)
                assigned = term_name in result.labels.get(cat_name, [])
                row.append(f"{score:.4f}")
                row.append("1" if assigned else "0")
            row.append(labels_to_caption(result))
            writer.writerow(row)


def write_report(
    term_list: TermList,
    results: list[ImageLabels],
    output_path: Path,
) -> None:
    """Write an HTML report with per-image labels, scores, and summary statistics."""
    h = html_mod.escape
    out = io.StringIO()
    out.write("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n")
    out.write("<title>labelman report</title>\n")
    out.write("<style>\n")
    out.write("body { font-family: sans-serif; margin: 2em; }\n")
    out.write("table { border-collapse: collapse; margin: 1em 0; }\n")
    out.write("th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }\n")
    out.write("th { background: #f0f0f0; }\n")
    out.write(".assigned { background: #d4edda; }\n")
    out.write(".suppressed { color: #999; }\n")
    out.write(".thumb { width: 48px; height: 48px; object-fit: cover; border-radius: 3px; vertical-align: middle; }\n")
    out.write("h2 { margin-top: 2em; }\n")
    out.write("</style>\n</head><body>\n")

    out.write("<h1>labelman Report</h1>\n")

    # --- Summary ---
    out.write("<h2>Summary</h2>\n")
    out.write(f"<p>Images: {len(results)}</p>\n")
    out.write(f"<p>Categories: {len(term_list.categories)}</p>\n")
    total_terms = sum(len(c.terms) for c in term_list.categories)
    out.write(f"<p>Terms: {total_terms}</p>\n")
    out.write(f"<p>Default threshold: {term_list.defaults.threshold}</p>\n")

    # --- Per-category statistics ---
    out.write("<h2>Category Statistics</h2>\n")
    for cat in term_list.categories:
        out.write(f"<h3>{h(cat.name)} <small>({cat.mode.value})</small></h3>\n")
        eff_thresh = cat.threshold if cat.threshold is not None else term_list.defaults.threshold
        out.write(f"<p>Effective threshold: {eff_thresh}</p>\n")

        # Count assignments per term
        term_counts: dict[str, int] = {t.term: 0 for t in cat.terms}
        no_assignment_count = 0
        for result in results:
            assigned = result.labels.get(cat.name, [])
            if not assigned:
                no_assignment_count += 1
            for t in assigned:
                if t in term_counts:
                    term_counts[t] += 1

        out.write("<table><tr><th>Term</th><th>Assigned count</th><th>% of images</th></tr>\n")
        for term in cat.terms:
            count = term_counts[term.term]
            pct = (count / len(results) * 100) if results else 0
            out.write(f"<tr><td>{h(term.term)}</td><td>{count}</td><td>{pct:.1f}%</td></tr>\n")
        if cat.mode != CategoryMode.EXACTLY_ONE:
            pct_none = (no_assignment_count / len(results) * 100) if results else 0
            out.write(f"<tr><td><em>(none)</em></td><td>{no_assignment_count}</td><td>{pct_none:.1f}%</td></tr>\n")
        out.write("</table>\n")

    # --- Per-image detail ---
    report_dir = output_path.parent.resolve()

    out.write("<h2>Per-Image Results</h2>\n")
    out.write("<table>\n<tr><th></th><th>Image</th><th>Caption</th>")
    col_specs = []
    for cat in term_list.categories:
        for t in cat.terms:
            col_specs.append((cat.name, t.term))
            out.write(f"<th>{h(cat.name)}/{h(t.term)}</th>")
    out.write("</tr>\n")

    for result in results:
        # Compute relative path from report to image
        try:
            rel = os.path.relpath(result.image, report_dir)
        except ValueError:
            rel = result.image
        rel_escaped = h(rel)
        img_tag = (
            f"<img src='{rel_escaped}' class='thumb' "
            f"onerror=\"this.style.display='none'\">"
        )
        out.write(f"<tr><td>{img_tag}</td>")
        out.write(f"<td>{h(result.image)}</td>")
        out.write(f"<td>{h(labels_to_caption(result))}</td>")
        for cat_name, term_name in col_specs:
            score = result.all_scores.get(cat_name, {}).get(term_name, 0.0)
            assigned = term_name in result.labels.get(cat_name, [])
            css = "assigned" if assigned else "suppressed"
            marker = " *" if assigned else ""
            out.write(f"<td class='{css}'>{score:.4f}{marker}</td>")
        out.write("</tr>\n")

    out.write("</table>\n")
    out.write("</body></html>\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out.getvalue())
