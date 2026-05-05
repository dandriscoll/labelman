"""CLI entrypoint for labelman."""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

from .check import check
from .errors import LabelmanError
from .integrations import get_descriptor, run_clip, run_qwen_vl, test_blip_endpoint, test_clip_endpoint, test_llm_endpoint, test_qwen_vl_endpoint
from .label import assemble_final_labels, load_manual_sidecar, merge_sidecars, write_csv, write_final_sidecar, write_report, write_sidecar
from .rename import rename_term
from .schema import parse
from .suggest import bootstrap, expand, format_suggest_result, write_suggest_sidecar
from .web import serve as web_serve


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

    summary = f"OK: {result.num_categories} categories, {result.num_terms} terms, {result.num_global_terms} global terms"
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
# llm:      Optional LLM integration (litellm-compatible) to refine
#           BLIP VQA answers into category labels. When configured,
#           vague BLIP answers are sent to the LLM with the category's
#           terms as options, and the LLM picks the best match.
#
# Run 'labelman descriptor blip' or 'labelman descriptor clip'
# to see the Boutiques descriptors for the built-in scripts.
integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify
    # script: /path/to/custom-clip.sh   # uncomment to use a custom script
  # llm:
  #   endpoint: http://localhost:11434/v1/chat/completions
  #   model: qwen3
  # qwen_vl:
  #   endpoint: http://localhost:8082/v1/chat/completions
  #   model: Qwen2.5-VL-7B-Instruct

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
  # question: semantic question — shown in suggest output and sent to the
  #           LLM for classification. Also used as the BLIP VQA prompt
  #           unless 'prompt' is set.
  # ask:      optional override for the BLIP VQA prompt. Use a simpler
  #           phrasing that BLIP handles well, while 'question' keeps the
  #           nuanced intent for the LLM. If omitted, 'question' is used
  #           for both. Can also be set per-term for yes/no probing.
  # question_answer_max_tokens: limit answer length (default: 100)
  - name: setting
    mode: zero-or-one
    threshold: 0.4
    question: "Is this photo taken indoors, outdoors, or in a studio?"
    ask: "Describe the setting of this photo."
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
        26: "Could not read a local file (check image path and permissions)",
        28: "Request timed out",
        35: "SSL/TLS connection error",
        52: "Server returned an empty response",
        55: "Failed to send request data",
        56: "Failed to receive response data",
    }

    # Extract image paths from command (everything after --images)
    images: list[str] = []
    cmd = e.cmd or []
    for i, arg in enumerate(cmd):
        if arg == "--images" and i + 1 < len(cmd):
            images = cmd[i + 1:]
            break

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

    # Show last few lines of stdout — sometimes errors land here
    if e.stdout:
        stdout_lines = e.stdout.strip().splitlines()
        # Show up to last 3 lines of stdout for context
        for line in stdout_lines[-3:]:
            lines.append(f"  [stdout] {line}")

    if images:
        lines.append(f"  Image: {images[0]}" if len(images) == 1
                     else f"  Images ({len(images)}): {images[0]} ... {images[-1]}")
    logger = logging.getLogger("labelman")
    logger.debug("integration error command: %s", " ".join(cmd))
    logger.debug("integration error stderr: %r", e.stderr)
    logger.debug("integration error stdout (last 500 chars): %s", (e.stdout or "")[-500:])

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

    start_at = getattr(args, "start_at", None)
    if start_at:
        skipped = _skip_until(image_paths, start_at)
        if skipped is None:
            print(f"Error: --start-at {start_at!r} did not match any image in {images_path}",
                  file=sys.stderr)
            return 1
        image_paths = image_paths[skipped:]
        print(f"Resuming from {Path(image_paths[0]).name} (skipped {skipped} earlier image(s))",
              flush=True)

    term_list = parse(config_path)
    try:
        sample = _parse_sample(args.sample, len(image_paths))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Build on_image callback to write sidecars incrementally
    dry_run = args.dry_run
    output_dir = Path(args.txt_output) if args.txt_output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    include_mined = args.include_mined
    written_count = 0

    def _on_image(suggestion, index, total):
        nonlocal written_count
        name = Path(suggestion.image).name
        if not dry_run:
            sidecar = write_suggest_sidecar(
                term_list, suggestion,
                include_mined=include_mined,
                output_dir=output_dir,
            )
            written_count += 1
            print(f"  [{index}/{total}] {name} -> {sidecar.name}", flush=True)
        else:
            print(f"  [{index}/{total}] {name}", flush=True)

    try:
        oaat = getattr(args, 'one_at_a_time', False)
        if args.mode == "bootstrap":
            result = bootstrap(term_list, image_paths, sample=sample,
                               one_at_a_time=oaat, on_image=_on_image)
        else:
            result = expand(term_list, image_paths, sample=sample,
                            one_at_a_time=oaat, on_image=_on_image)
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

    if written_count:
        print(f"Wrote {written_count} sidecar(s)")

    return 0


def _find_images(directory: Path) -> list[str]:
    """Find image files in a directory, sorted by name."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sorted(
        str(p) for p in directory.iterdir()
        if p.suffix.lower() in extensions
    )


def _skip_until(image_paths: list[str], marker: str) -> int | None:
    """Return the index of the first image matching `marker`, or None.

    Matching order: exact full-path match, then basename match. Useful for
    resuming a run — the user can pass either the filename printed in the
    progress log or the full path.
    """
    for i, p in enumerate(image_paths):
        if p == marker or Path(p).name == marker:
            return i
    return None


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


def _resolve_provider(args: argparse.Namespace, term_list) -> str:
    """Determine which provider to use for labeling."""
    provider = getattr(args, "provider", None)
    if provider:
        return provider
    # Auto-detect: prefer qwen_vl if configured, fall back to clip
    if term_list.integrations.qwen_vl is not None:
        return "qwen_vl"
    return "clip"


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

    provider = _resolve_provider(args, term_list)

    # Validate provider config
    if provider == "qwen_vl" and term_list.integrations.qwen_vl is None:
        print("Error: --provider qwen_vl requires integrations.qwen_vl in labelman.yaml", file=sys.stderr)
        return 1
    if provider == "clip" and term_list.integrations.clip is None:
        print("Error: --provider clip requires integrations.clip in labelman.yaml", file=sys.stderr)
        return 1

    num_images = len(image_paths)
    num_cats = len(term_list.categories)
    if not quiet:
        print(f"Labeling {num_images} images against {total_terms} terms in {num_cats} categories "
              f"(provider: {provider})...", flush=True)

    # Determine output directory
    output_dir = Path(args.output) if args.output else images_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Score and label each image
    results = []
    for i, image_path in enumerate(image_paths, 1):
        name = Path(image_path).name

        try:
            if provider == "qwen_vl":
                vlm_labels = run_qwen_vl(term_list, image_path)
                manual = load_manual_sidecar(image_path)
                result = assemble_final_labels(
                    term_list, image_path, scores={},
                    manual_labels=manual, vlm_labels=vlm_labels,
                )
            else:
                clip_results = run_clip(term_list, [image_path])
                scores = _reshape_clip_scores(term_list, clip_results).get(image_path, {})
                manual = load_manual_sidecar(image_path)
                result = assemble_final_labels(term_list, image_path, scores, manual_labels=manual)
        except subprocess.CalledProcessError as e:
            for line in _integration_error_message(e):
                print(line, file=sys.stderr)
            return 1

        results.append(result)

        # Write per-image detected sidecar
        sidecar = write_sidecar(result, output_dir=output_dir)
        if not quiet:
            n_detected = sum(len(v) for v in result.labels.values())
            print(f"  [{i}/{num_images}] {name} -> {sidecar.name} ({n_detected} detected)", flush=True)

    # Write CSV
    csv_path = output_dir / "labels.csv"
    write_csv(term_list, results, csv_path)

    # Write HTML report
    report_path = output_dir / "report.html"
    write_report(term_list, results, report_path)

    if not quiet:
        print(f"Done: {len(results)} images labeled")
        print(f"  Sidecars: {output_dir}/*.detected.txt")
        print(f"  CSV:      {csv_path}")
        print(f"  Report:   {report_path}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
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
    output_dir = Path(args.output) if args.output else None

    count = 0
    for image_path in image_paths:
        merged = merge_sidecars(term_list, image_path)
        sidecar = write_final_sidecar(image_path, merged, output_dir=output_dir)
        count += 1
        name = Path(image_path).name
        print(f"  {name} -> {sidecar.name} ({len(merged)} labels)")

    print(f"Applied {count} image(s)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    images_path = Path(args.images)
    if not images_path.is_dir():
        print(f"Error: {images_path} is not a directory", file=sys.stderr)
        return 1

    web_serve(images_path, host=args.host, port=args.port,
              markback=args.markback)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .build import build_yaml

    images_path = Path(args.images)
    if not images_path.is_dir():
        print(f"Error: {images_path} is not a directory", file=sys.stderr)
        return 1

    config_path = images_path / DEFAULT_CONFIG
    if config_path.exists() and not args.force:
        print(f"Error: {config_path} already exists (use --force to overwrite)",
              file=sys.stderr)
        return 1

    yaml_text, terms, files_scanned = build_yaml(images_path)
    if not terms:
        print(f"Error: no labels found in {images_path} "
              f"(looked for *.txt and *.mb final-label sidecars)",
              file=sys.stderr)
        return 1

    # Defense in depth: parse what we just generated. parse() raises
    # ParseError (a LabelmanError) on failure, which the top-level handler
    # surfaces cleanly.
    parse(yaml_text)

    config_path.write_text(yaml_text)
    print(f"Wrote {config_path}: {len(terms)} unique term(s) from "
          f"{files_scanned} sidecar file(s)")
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    result = rename_term(
        config_path=config_path,
        old=args.old,
        new=args.new,
        dry_run=args.dry_run,
    )

    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    prefix = "[dry run] " if args.dry_run else ""
    for change in result.changes:
        print(f"  {prefix}{change}")
    print(f"{prefix}Renamed '{args.old}' -> '{args.new}' ({len(result.changes)} file(s))")
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


def cmd_test_endpoints(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: {config_path} not found", file=sys.stderr)
        return 1

    term_list = parse(config_path)
    integ = term_list.integrations

    # Build list of (name, url, test_fn) tuples
    tests: list[tuple[str, str, callable]] = []
    if integ.blip and integ.blip.endpoint:
        tests.append(("blip", integ.blip.endpoint,
                       lambda: test_blip_endpoint(integ.blip.endpoint)))
    if integ.clip and integ.clip.endpoint:
        tests.append(("clip", integ.clip.endpoint,
                       lambda: test_clip_endpoint(integ.clip.endpoint)))
    if integ.llm:
        tests.append(("llm", integ.llm.endpoint,
                       lambda: test_llm_endpoint(integ.llm)))
    if integ.qwen_vl:
        tests.append(("qwen_vl", integ.qwen_vl.endpoint,
                       lambda: test_qwen_vl_endpoint(integ.qwen_vl)))

    if not tests:
        print("No endpoints configured in labelman.yaml")
        return 0

    any_failed = False
    for name, url, test_fn in tests:
        print(f"  {name:10s} {url} ... ", end="", flush=True)
        ok, message = test_fn()
        print(message)
        if not ok:
            any_failed = True

    return 1 if any_failed else 0


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
    suggest_p.add_argument("--dry-run", action="store_true",
                           help="Print suggestions without writing .txt sidecar files")
    suggest_p.add_argument("--txt-output", default=None,
                           help="Output directory for .txt sidecars (default: same as --images)")
    suggest_p.add_argument("--include-mined", action="store_true",
                           help="Include mined terms in .txt sidecars (default: only existing category labels)")
    suggest_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")
    suggest_p.add_argument("--start-at", default=None,
                           help="Skip images before this filename (basename or full path). "
                                "Useful for resuming an interrupted run.")
    suggest_p.add_argument("--one-at-a-time", action="store_true",
                           help="Send images to BLIP/CLIP one at a time (slower, but pinpoints errors)")

    # label
    label_p = sub.add_parser("label", help="Apply taxonomy labels to images")
    label_p.add_argument("--images", default=".", help="Directory containing images")
    label_p.add_argument("--output", default=None,
                         help="Output directory for sidecars, CSV, and report (default: same as --images)")
    label_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")
    label_p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    label_p.add_argument("--provider", choices=["clip", "qwen_vl"], default=None,
                         help="Classification provider (default: auto-detect from config, prefers qwen_vl)")

    # apply
    apply_p = sub.add_parser("apply", help="Merge .labels.txt + .detected.txt into final .txt sidecars")
    apply_p.add_argument("--images", default=".", help="Directory containing images")
    apply_p.add_argument("--output", default=None,
                         help="Output directory for .txt sidecars (default: same as --images)")
    apply_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")

    # ui
    ui_p = sub.add_parser("ui", help="Launch web-based labeling interface")
    ui_p.add_argument("--images", default=".", help="Directory containing images")
    ui_p.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    ui_p.add_argument("--port", type=int, default=7933, help="Port to bind to (default: 7933)")
    ui_p.add_argument("--markback", action="store_true",
                      help="Read/write sidecars as markback v2 (.mb / .labels.mb / "
                           ".detected.mb with ';' separator) instead of .txt with ','")

    # build
    build_p = sub.add_parser("build", help="Generate labelman.yaml from existing .txt/.mb sidecars")
    build_p.add_argument("--images", default=".", help="Directory containing images and sidecars")
    build_p.add_argument("--force", action="store_true",
                         help="Overwrite an existing labelman.yaml")

    # rename
    rename_p = sub.add_parser("rename", help="Rename a term across config and sidecar files")
    rename_p.add_argument("old", help="Current term name")
    rename_p.add_argument("new", help="New term name")
    rename_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")
    rename_p.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")

    # descriptor
    desc_p = sub.add_parser("descriptor", help="Print a Boutiques descriptor for a built-in integration")
    desc_p.add_argument("tool", choices=["blip", "clip"], help="Integration name")

    # test-endpoints
    test_p = sub.add_parser("test-endpoints", help="Test connectivity to configured endpoints")
    test_p.add_argument("--config", default=DEFAULT_CONFIG, help="Path to labelman.yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    # Pull --verbose/-v out of argv so it works in any position,
    # even before the subcommand where argparse can't see it.
    if argv is None:
        argv = sys.argv[1:]
    verbose = False
    cleaned: list[str] = []
    for arg in argv:
        if arg in ("--verbose", "-v"):
            verbose = True
        else:
            cleaned.append(arg)

    args = parser.parse_args(cleaned)
    args.verbose = verbose

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s: %(message)s",
            stream=sys.stderr,
        )

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "init": cmd_init,
        "check": cmd_check,
        "suggest": cmd_suggest,
        "label": cmd_label,
        "apply": cmd_apply,
        "rename": cmd_rename,
        "ui": cmd_ui,
        "build": cmd_build,
        "descriptor": cmd_descriptor,
        "test-endpoints": cmd_test_endpoints,
    }
    try:
        return handlers[args.command](args)
    except LabelmanError as e:
        # All user-facing errors (config parse, integration failures, UI
        # bind errors, …) land here as clean single-/multi-line messages.
        for line in str(e).splitlines() or [""]:
            print(f"Error: {line}" if line and not line.startswith(" ") else line,
                  file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
