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

    print(f"OK: {result.num_categories} categories, {result.num_terms} terms")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / DEFAULT_CONFIG
    if config_path.exists() and not args.force:
        print(f"Error: {config_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    starter = """\
defaults:
  threshold: 0.3

integrations:
  blip:
    endpoint: http://localhost:8080/caption
  clip:
    endpoint: http://localhost:8081/classify

categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
      - term: object

  - name: style
    mode: zero-or-more
    terms:
      - term: photographic
      - term: illustration
      - term: abstract
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
