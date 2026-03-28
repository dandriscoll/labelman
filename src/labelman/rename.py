"""Rename a term across the YAML config and all sidecar txt files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .schema import parse


@dataclass
class RenameResult:
    changes: list[str] = field(default_factory=list)
    error: str | None = None


def _rename_in_yaml(text: str, old: str, new: str) -> str | None:
    """Replace a term name in labelman.yaml text, preserving formatting.

    Returns the updated text, or None if no changes were made.
    """
    changed = False
    lines = text.split("\n")
    result = []
    for line in lines:
        # Match "term: old" (with optional quotes) in category term entries
        # Handles: "- term: foo", "  term: foo", "  - term: 'foo'", etc.
        m = re.match(r'^(\s*-?\s*term:\s*)(["\']?)' + re.escape(old) + r'(["\']?\s*)$', line)
        if m:
            result.append(m.group(1) + m.group(2) + new + m.group(3))
            changed = True
            continue
        # Match "- old" in global_terms list (bare string entries)
        # Be careful: only match when it looks like a list item with exactly the old term
        m = re.match(r'^(\s*-\s+)(["\']?)' + re.escape(old) + r'(["\']?\s*)$', line)
        if m:
            result.append(m.group(1) + m.group(2) + new + m.group(3))
            changed = True
            continue
        result.append(line)
    if not changed:
        return None
    return "\n".join(result)


def _rename_in_sidecar(path: Path, old: str, new: str) -> bool:
    """Replace a term in a comma-separated sidecar file.

    Returns True if the file was modified.
    """
    text = path.read_text().strip()
    if not text:
        return False
    labels = [label.strip() for label in text.split(",")]
    changed = False
    updated = []
    for label in labels:
        # Handle suppression prefix
        if label == old or label == f"-{old}":
            prefix = "-" if label.startswith("-") else ""
            updated.append(f"{prefix}{new}")
            changed = True
        else:
            updated.append(label)
    if not changed:
        return False
    path.write_text(", ".join(updated))
    return True


def rename_term(
    config_path: Path,
    old: str,
    new: str,
    dry_run: bool = False,
) -> RenameResult:
    """Rename a term in the YAML config and all sidecar txt files.

    Sidecar files are found by scanning the config file's parent directory
    for .labels.txt, .detected.txt, and .txt files.

    Args:
        config_path: Path to labelman.yaml.
        old: Current term name.
        new: New term name.
        dry_run: If True, report changes without writing files.

    Returns:
        RenameResult with list of changes made (or that would be made).
    """
    result = RenameResult()

    if old == new:
        result.error = "old and new term names are the same"
        return result

    # Validate: parse the config to confirm the old term exists
    term_list = parse(config_path)
    found = False
    for cat in term_list.categories:
        for t in cat.terms:
            if t.term == old:
                found = True
                break
    if not found and old in term_list.global_terms:
        found = True
    if not found:
        result.error = f"term '{old}' not found in {config_path}"
        return result

    # Check that new term doesn't already exist (would create duplicates)
    for cat in term_list.categories:
        for t in cat.terms:
            if t.term == new:
                result.error = f"term '{new}' already exists in category '{cat.name}'"
                return result
    if new in term_list.global_terms:
        result.error = f"term '{new}' already exists in global_terms"
        return result

    # Rename in YAML
    yaml_text = config_path.read_text()
    updated_yaml = _rename_in_yaml(yaml_text, old, new)
    if updated_yaml is not None:
        if not dry_run:
            config_path.write_text(updated_yaml)
        result.changes.append(str(config_path))

    # Rename in sidecar files in the config's directory
    base_dir = config_path.parent
    for suffix in (".labels.txt", ".detected.txt", ".txt"):
        for path in sorted(base_dir.glob(f"*{suffix}")):
            # Skip .detected.txt and .labels.txt when looking for plain .txt
            if suffix == ".txt" and (
                path.name.endswith(".labels.txt")
                or path.name.endswith(".detected.txt")
            ):
                continue
            text = path.read_text().strip()
            if not text:
                continue
            labels = [label.strip() for label in text.split(",")]
            # Check if old term (or -old suppression) is present
            if old not in labels and f"-{old}" not in labels:
                continue
            if dry_run:
                result.changes.append(str(path))
            else:
                if _rename_in_sidecar(path, old, new):
                    result.changes.append(str(path))

    return result
