"""Integration resolution and invocation for BLIP/CLIP/LLM."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .schema import IntegrationConfig, LLMConfig, TermList

logger = logging.getLogger("labelman")

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
    one_at_a_time: bool = False,
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

    if not all_terms:
        logger.debug("clip: no terms to score, returning empty scores")
        return [{"image": p, "scores": {}} for p in image_paths]

    terms_str = ",".join(all_terms)

    base_cmd = [script]
    if endpoint:
        base_cmd += ["--endpoint", endpoint]
    base_cmd += ["--terms", terms_str]

    logger.debug("clip: scoring %d image(s) against %d term(s)", len(image_paths), len(all_terms))

    batches = [[p] for p in image_paths] if one_at_a_time else [image_paths]
    output = []
    for batch in batches:
        cmd = base_cmd + ["--images"] + batch
        logger.debug("clip: %s", Path(batch[0]).name if len(batch) == 1
                     else f"{len(batch)} image(s)")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().splitlines():
            entry = json.loads(line)
            output.append(entry)
            logger.debug("clip: %s -> %s", Path(entry["image"]).name, entry.get("scores", {}))
    return output


def run_llm(
    config: LLMConfig,
    system_prompt: str,
    user_message: str,
) -> str:
    """Call a litellm-compatible chat completions endpoint.

    Returns the assistant's response text.
    """
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if config.model:
        body["model"] = config.model

    logger.debug("llm: POST %s (model=%s)", config.endpoint, config.model or "default")
    logger.debug("llm: system: %s", system_prompt)
    logger.debug("llm: user: %s", user_message)

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        config.endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        logger.debug("llm: HTTP %d: %s", e.code, error_body)
        raise RuntimeError(
            f"LLM request failed (HTTP {e.code}): {error_body[:200]}"
        ) from e

    raw_content = result["choices"][0]["message"]["content"]
    # Strip <think>...</think> blocks from reasoning models (e.g. qwen3)
    content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content)
    content = content.strip()
    if raw_content != content:
        logger.debug("llm: raw response (before stripping think tags): %s", raw_content[:200])
    logger.debug("llm: response: %s", content)
    return content


def run_blip(
    term_list: TermList,
    image_paths: list[str],
    prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    one_at_a_time: bool = False,
) -> list[dict]:
    """Invoke the BLIP integration and return per-image captions.

    Returns:
        List of dicts: [{"image": str, "caption": str}, ...]
    """
    config = term_list.integrations.blip
    script, endpoint = resolve_script("blip", config)

    base_cmd = [script]
    if endpoint:
        base_cmd += ["--endpoint", endpoint]
    if prompt:
        base_cmd += ["--prompt", prompt]
    if max_tokens is not None:
        base_cmd += ["--max-tokens", str(max_tokens)]

    logger.debug("blip: captioning %d image(s)%s", len(image_paths),
                 f" with prompt: {prompt}" if prompt else "")

    batches = [[p] for p in image_paths] if one_at_a_time else [image_paths]
    output = []
    for batch in batches:
        cmd = base_cmd + ["--images"] + batch
        logger.debug("blip: %s", Path(batch[0]).name if len(batch) == 1
                     else f"{len(batch)} image(s)")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.strip().splitlines():
            entry = json.loads(line)
            output.append(entry)
            text = entry.get("answer") or entry.get("caption") or ""
            logger.debug("blip: %s -> %s", Path(entry["image"]).name, text)
    return output
