"""CLI entrypoint for labelman."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .check import check
from .integrations import get_descriptor


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
  - name: setting
    mode: zero-or-one
    threshold: 0.4
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
        "descriptor": cmd_descriptor,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
