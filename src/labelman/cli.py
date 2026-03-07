"""CLI entrypoint for labelman."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .check import check
from .integrations import get_descriptor, run_clip
from .label import assemble_final_labels, load_manual_sidecar, write_csv, write_report, write_sidecar
from .schema import parse
from .suggest import bootstrap, expand, format_suggest_result


DEFAULT_CONFIG = "labelman.yaml"


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if not path.is_file():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    result = check(str(path))

    if not result.ok:
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    for warn in result.warnings:
        print(f"WARNING: {warn}", file=sys.stderr)

    summary = f"OK: {result.num_categories} categories, {result.num_terms} terms"
    if result.num_global_terms:
        summary += f", {result.num_global_terms} global terms"
    print(summary)
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / DEFAULT_CONFIG
    if config_path.exists() and not args.force:
        print(f"Error: {config_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    starter = """\
# labelman configuration
# See: docs/DESIGN.md for full reference

# --- Global defaults ---
# threshold: minimum confidence score for a label to be applied.
# Can be overridden per-category or per-term.
defaults:
  threshold: 0.3

# --- Global baseline terms ---
# Labels applied to every image, bypassing detection and thresholds.
# Useful for dataset-wide tags like aircraft type or project identifier.
# global_terms:
#   - aircraft
#   - mooney m20

# --- Integrations ---
# Configure how labelman calls BLIP (captioning) and CLIP (classification).
#
# endpoint: URL of a running BLIP/CLIP HTTP service.
#           labelman uses its built-in scripts to call these.
#
# script:   Path to a custom script. When set, the built-in script
#           is bypassed and endpoint is ignored.
#
# Run 'labelman descriptor blip' or 'labelman descriptor clip'
# to see the Boutiques descriptors for the built-in scripts.
integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify
    # script: /path/to/custom-clip.sh   # uncomment to use a custom script

# --- Categories ---
# Each category is a labeling axis. The mode controls how many labels
# from this category can be assigned to a single image:
#
#   exactly-one  : always assigns the top-scoring term (forced)
#   zero-or-one  : assigns the top-scoring term only if it meets threshold
#   zero-or-more : assigns all terms that meet threshold

categories:
  # exactly-one: every image gets exactly one label from this category.
  # If no term meets the threshold, the highest-scoring term is still assigned.
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
      - term: object
      - term: landscape

  # zero-or-one: at most one label. If nothing scores above threshold, none is assigned.
  # threshold here overrides the global default for this category.
  # question: optional VQA prompt for BLIP during 'suggest'
  # question_answer_max_tokens: limit answer length (default: 100)
  - name: setting
    mode: zero-or-one
    threshold: 0.4
    question: "Is this photo taken indoors, outdoors, or in a studio?"
    question_answer_max_tokens: 10
    terms:
      - term: indoor
      - term: outdoor
      - term: studio

  # zero-or-more: any number of labels can be assigned.
  # Individual terms can override the threshold for fine-grained control.
  - name: style
    mode: zero-or-more
    terms:
      - term: photographic
      - term: illustration
      - term: abstract
        threshold: 0.5    # higher threshold — only assign if confident
"""
    config_path.write_text(starter)
    print(f"Created {config_path}")
    return 0


def _integration_error_message(e: subprocess.CalledProcessError) -> list[str]:
    """Turn a CalledProcessError from an integration script into useful messages."""
    cmd_name = Path(e.cmd[0]).name if e.cmd else "unknown"
    tool = cmd_name.replace(".sh", "")  # "blip.sh" -> "blip"
    endpoint = None
    for i, arg in enumerate(e.cmd or []):
        if arg == "--endpoint" and i + 1 < len(e.cmd):
            endpoint = e.cmd[i + 1]
            break

    # These exit codes come from curl, which the integration scripts wrap
    curl_errors = {
        6:  "Could not resolve hostname",
        7:  "Connection refused",
        22: "Server returned an HTTP error",
        28: "Request timed out",
        35: "SSL/TLS connection error",
        52: "Server returned an empty response",
        55: "Failed to send request data",
        56: "Failed to receive response data",
    }

    lines = []
    reason = curl_errors.get(e.returncode)
    if reason and endpoint:
        lines.append(f"Error: {reason} — {endpoint}")
        lines.append(f"  Check that the {tool} service is running and the endpoint in labelman.yaml is correct.")
    elif reason:
        lines.append(f"Error: {reason} (from {cmd_name})")
        lines.append(f"  Check the integrations.{tool} section in labelman.yaml.")
    else:
        lines.append(f"Error: {cmd_name} exited with code {e.returncode}")

    if e.stderr:
        for line in e.stderr.strip().splitlines():
            lines.append(f"  {line}")

    return lines


def _parse_sample(value: str | None, total: int) -> int | None:
    """Parse a sample value like '10' or '25%' into an absolute count."""
    if value is None:
        return None
    value = value.strip()
    if value.endswith("%"):
        try:
            pct = float(value[:-1])
        except ValueError:
            raise ValueError(f"Invalid percentage: {value}")
        if not (0 < pct <= 100):
            raise ValueError(f"Percentage must be between 0 and 100, got {value}")
        return max(1, math.ceil(total * pct / 100))
    else:
        try:
            n = int(value)
        except ValueError:
            raise ValueError(f"Invalid sample value: {value} (use a number or percentage like 25%)")
        if n < 1:
            raise ValueError(f"Sample count must be at least 1, got {n}")
        return n


def cmd_suggest(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    images_path = Path(args.images)
    if not images_path.is_dir():
        print(f"Error: {images_path} is not a directory", file=sys.stderr)
        return 1

    image_paths = _find_images(images_path)
    if not image_paths:
        print(f"Error: no images found in {images_path}", file=sys.stderr)
        return 1

    term_list = parse(config_path)
    try:
        sample = _parse_sample(args.sample, len(image_paths))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        if args.mode == "bootstrap":
            result = bootstrap(term_list, image_paths, sample=sample)
        else:
            result = expand(term_list, image_paths, sample=sample)
    except subprocess.CalledProcessError as e:
        for line in _integration_error_message(e):
            print(line, file=sys.stderr)
        return 1

    output = format_suggest_result(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Wrote suggestions to {out_path}")
    else:
        print(output, end="")

    return 0


def _find_images(directory: Path) -> list[str]:
    """Find image files in a directory, sorted by name."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sorted(
        str(p) for p in directory.iterdir()
        if p.suffix.lower() in extensions
    )


def _reshape_clip_scores(term_list, clip_results: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Reshape flat CLIP scores into per-category scores keyed by image path.

    CLIP returns: [{"image": path, "scores": {"term": float, ...}}]
    apply_labels expects: {"category": {"term": float, ...}}
    """
    # Build term->category mapping
    term_to_cat: dict[str, str] = {}
    for cat in term_list.categories:
        for t in cat.terms:
            term_to_cat[t.term] = cat.name

    result = {}
    for entry in clip_results:
        image = entry["image"]
        flat_scores = entry.get("scores", {})
        cat_scores: dict[str, dict[str, float]] = {}
        for term, score in flat_scores.items():
            cat_name = term_to_cat.get(term)
            if cat_name is not None:
                if cat_name not in cat_scores:
                    cat_scores[cat_name] = {}
                cat_scores[cat_name][term] = score
        result[image] = cat_scores
    return result


def cmd_label(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    images_path = Path(args.images)
    if not images_path.is_dir():
        print(f"Error: {images_path} is not a directory", file=sys.stderr)
        return 1

    image_paths = _find_images(images_path)
    if not image_paths:
        print(f"Error: no images found in {images_path}", file=sys.stderr)
        return 1

    quiet = args.quiet
    term_list = parse(config_path)

    # Check that there are terms to label with
    total_terms = sum(len(c.terms) for c in term_list.categories)
    if total_terms == 0:
        print("Error: no terms defined in any category — nothing to label", file=sys.stderr)
        return 1

    num_images = len(image_paths)
    num_cats = len(term_list.categories)
    if not quiet:
        print(f"Scoring {num_images} images against {total_terms} terms in {num_cats} categories...")

    # Run CLIP
    try:
        clip_results = run_clip(term_list, image_paths)
    except subprocess.CalledProcessError as e:
        for line in _integration_error_message(e):
            print(line, file=sys.stderr)
        return 1

    scores_by_image = _reshape_clip_scores(term_list, clip_results)

    # Determine output directory
    output_dir = Path(args.output) if args.output else images_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Label each image
    if not quiet:
        print("Labeling images...")
    results = []
    for i, image_path in enumerate(image_paths, 1):
        scores = scores_by_image.get(image_path, {})
        manual = load_manual_sidecar(image_path)
        result = assemble_final_labels(term_list, image_path, scores, manual_labels=manual)
        results.append(result)

        # Write per-image sidecar
        sidecar = write_sidecar(result, output_dir=output_dir)
        if not quiet:
            name = Path(image_path).name
            n_labels = len(result.final_labels)
            print(f"  [{i}/{num_images}] {name} -> {sidecar.name} ({n_labels} labels)")

    # Write CSV
    csv_path = output_dir / "labels.csv"
    write_csv(term_list, results, csv_path)

    # Write HTML report
    report_path = output_dir / "report.html"
    write_report(term_list, results, report_path)

    if not quiet:
        print(f"Done: {len(results)} images labeled")
        print(f"  Sidecars: {output_dir}/*.txt")
        print(f"  CSV:      {csv_path}")
        print(f"  Report:   {report_path}")
    return 0


def cmd_descriptor(args: argparse.Namespace) -> int:
    tool = args.tool
    try:
        desc = get_descriptor(tool)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(desc, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labelman", description="Bulk image labeling tool")
    sub = parser.add_subparsers(dest="command")

    # init
    init_p = sub.add_parser("init", help="Initialize a labeling workspace")
    init_p.add_argument("--dir", default=".", help="Target directory (default: current)")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing labelman.yaml")

    # check
    check_p = sub.add_parser("check", help="Validate labelman.yaml")
    check_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")

    # suggest
    suggest_p = sub.add_parser("suggest", help="Propose taxonomy terms from image analysis")
    suggest_p.add_argument("--mode", choices=["bootstrap", "expand"], default="bootstrap",
                           help="bootstrap: propose new categories/terms; expand: add terms to existing categories")
    suggest_p.add_argument("--sample", default=None,
                           help="Number (e.g. 10) or percentage (e.g. 25%%) of images to sample")
    suggest_p.add_argument("--images", default=".", help="Directory containing images")
    suggest_p.add_argument("--output", default=None, help="Write suggestions to file (default: stdout)")
    suggest_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")

    # label
    label_p = sub.add_parser("label", help="Apply taxonomy labels to images")
    label_p.add_argument("--images", default=".", help="Directory containing images")
    label_p.add_argument("--output", default=None,
                         help="Output directory for sidecars, CSV, and report (default: same as --images)")
    label_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")
    label_p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    # descriptor
    desc_p = sub.add_parser("descriptor", help="Print a Boutiques descriptor for a built-in integration")
    desc_p.add_argument("tool", choices=["blip", "clip"], help="Integration name")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "init": cmd_init,
        "check": cmd_check,
        "suggest": cmd_suggest,
        "label": cmd_label,
        "descriptor": cmd_descriptor,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
