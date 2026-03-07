"""Integration resolution and invocation for BLIP/CLIP."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from .schema import IntegrationConfig, TermList

_PACKAGE_DIR = Path(__file__).parent
_SCRIPTS_DIR = _PACKAGE_DIR / "scripts"
_DESCRIPTORS_DIR = _PACKAGE_DIR / "descriptors"


def get_descriptor_path(tool: str) -> Path:
    """Return the path to a bundled Boutiques descriptor.

    Args:
        tool: One of 'blip' or 'clip'.

    Raises:
        ValueError: If the tool name is not recognized.
    """
    path = _DESCRIPTORS_DIR / f"{tool}.boutiques.json"
    if not path.is_file():
        raise ValueError(f"Unknown integration: {tool!r}. Available: blip, clip")
    return path


def get_descriptor(tool: str) -> dict:
    """Load and return a bundled Boutiques descriptor as a dict."""
    path = get_descriptor_path(tool)
    return json.loads(path.read_text())


def _builtin_script(tool: str) -> Path:
    return _SCRIPTS_DIR / f"{tool}.sh"


def resolve_script(tool: str, config: Optional[IntegrationConfig]) -> tuple[str, Optional[str]]:
    """Resolve which script and endpoint to use for a given integration.

    Returns:
        (script_path, endpoint_or_none)
        - If config.script is set, returns (config.script, None).
        - If config.endpoint is set, returns (builtin_script, config.endpoint).
        - If config is None, raises.
    """
    if config is None:
        raise ValueError(
            f"Integration '{tool}' is not configured in labelman.yaml. "
            f"Add an integrations.{tool} section with 'endpoint' or 'script'."
        )
    if config.script:
        return config.script, None
    assert config.endpoint is not None
    return str(_builtin_script(tool)), config.endpoint


def run_clip(
    term_list: TermList,
    image_paths: list[str],
) -> list[dict]:
    """Invoke the CLIP integration and return per-image scores.

    Returns:
        List of dicts: [{"image": str, "scores": {"term": float, ...}}, ...]
    """
    config = term_list.integrations.clip
    script, endpoint = resolve_script("clip", config)

    all_terms = []
    for cat in term_list.categories:
        for t in cat.terms:
            all_terms.append(t.term)
    terms_str = ",".join(all_terms)

    cmd = [script]
    if endpoint:
        cmd += ["--endpoint", endpoint]
    cmd += ["--terms", terms_str, "--images"] + image_paths

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    output = []
    for line in result.stdout.strip().splitlines():
        output.append(json.loads(line))
    return output


def run_blip(
    term_list: TermList,
    image_paths: list[str],
    prompt: Optional[str] = None,
) -> list[dict]:
    """Invoke the BLIP integration and return per-image captions.

    Returns:
        List of dicts: [{"image": str, "caption": str}, ...]
    """
    config = term_list.integrations.blip
    script, endpoint = resolve_script("blip", config)

    cmd = [script]
    if endpoint:
        cmd += ["--endpoint", endpoint]
    if prompt:
        cmd += ["--prompt", prompt]
    cmd += ["--images"] + image_paths

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    output = []
    for line in result.stdout.strip().splitlines():
        output.append(json.loads(line))
    return output
