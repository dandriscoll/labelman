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

import base64

from .schema import Category, CategoryMode, IntegrationConfig, LLMConfig, QwenVLConfig, TermList

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


def _encode_image(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, media_type)."""
    p = Path(image_path)
    suffix = p.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return data, media_type


def _build_category_spec(cat: Category) -> list[str]:
    """Return the list of allowed term strings for a category."""
    return [t.term for t in cat.terms]


def _build_qwen_vl_system_prompt(term_list: TermList) -> str:
    """Build the system prompt with the allowed terms per category.

    Includes category mode constraints so the model understands
    exactly-one vs zero-or-more semantics.
    """
    _mode_desc = {
        CategoryMode.EXACTLY_ONE: "Pick exactly one term (required).",
        CategoryMode.ZERO_OR_ONE: "Pick at most one term, or null if none apply.",
        CategoryMode.ZERO_OR_MORE: "Pick all terms that apply, or an empty list if none.",
    }
    taxonomy: dict[str, dict] = {}
    for cat in term_list.categories:
        taxonomy[cat.name] = {
            "terms": _build_category_spec(cat),
            "rule": _mode_desc[cat.mode],
        }

    lines = [
        "You are an image classifier. Classify the image into the categories below. "
        "Each category lists its allowed terms and a selection rule. "
        "Select only from the allowed terms and follow each category's rule strictly.",
        "",
        json.dumps(taxonomy, indent=2),
        "",
        "Respond with a JSON object mapping each category name to the selected "
        "term(s). For categories requiring exactly one term, use a string. "
        "For categories allowing multiple, use a list. Use null for no match.",
    ]
    return "\n".join(lines)


def _build_qwen_vl_user_prompt() -> str:
    """Build the user-turn text that accompanies the image."""
    return "Classify this image. Respond with JSON only."


def run_qwen_vl(
    term_list: TermList,
    image_path: str,
) -> dict[str, list[str]]:
    """Classify a single image using Qwen2.5-VL with category constraints.

    Unlike CLIP (which returns scores), Qwen2.5-VL understands the category
    mode constraints directly and returns label selections per category.

    Returns:
        Dict mapping category name to list of selected term strings.
        For exactly-one/zero-or-one modes, the list has 0 or 1 elements.
        For zero-or-more, the list may have any number of elements.
    """
    config = term_list.integrations.qwen_vl
    if config is None:
        raise ValueError(
            "Integration 'qwen_vl' is not configured in labelman.yaml. "
            "Add an integrations.qwen_vl section with 'endpoint' and 'model'."
        )

    system_prompt = _build_qwen_vl_system_prompt(term_list)
    user_text = _build_qwen_vl_user_prompt()
    img_data, media_type = _encode_image(image_path)

    body = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{img_data}",
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ],
        "max_tokens": 512,
    }

    logger.debug("qwen_vl: POST %s (model=%s) image=%s",
                 config.endpoint, config.model, Path(image_path).name)
    logger.debug("qwen_vl: system prompt: %s", system_prompt[:300])

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
        logger.debug("qwen_vl: HTTP %d: %s", e.code, error_body)
        raise RuntimeError(
            f"Qwen VL request failed (HTTP {e.code}): {error_body[:200]}"
        ) from e

    raw_content = result["choices"][0]["message"]["content"]
    # Strip <think>...</think> blocks if present
    content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    logger.debug("qwen_vl: raw response: %s", raw_content[:300])
    logger.debug("qwen_vl: parsed content: %s", content[:300])

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("qwen_vl: failed to parse JSON response: %s", content[:200])
        raise RuntimeError(
            f"Qwen VL returned invalid JSON: {content[:200]}"
        ) from e

    # Validate and normalize the response against the taxonomy
    valid_terms: dict[str, set[str]] = {}
    for cat in term_list.categories:
        valid_terms[cat.name] = {t.term for t in cat.terms}

    labels: dict[str, list[str]] = {}
    for cat in term_list.categories:
        raw_value = parsed.get(cat.name)
        if raw_value is None:
            labels[cat.name] = []
            continue

        # Normalize to list
        if isinstance(raw_value, str):
            selected = [raw_value] if raw_value else []
        elif isinstance(raw_value, list):
            selected = [str(v) for v in raw_value if v]
        else:
            selected = [str(raw_value)]

        # Filter to valid terms only
        filtered = [t for t in selected if t in valid_terms.get(cat.name, set())]

        # Enforce category constraints
        if cat.mode == CategoryMode.EXACTLY_ONE:
            if filtered:
                labels[cat.name] = [filtered[0]]
            elif cat.terms:
                # Model failed to pick one — fall back to first term
                labels[cat.name] = [cat.terms[0].term]
                logger.warning("qwen_vl: %s exactly-one fallback to %r for %s",
                               cat.name, cat.terms[0].term, Path(image_path).name)
            else:
                labels[cat.name] = []
        elif cat.mode == CategoryMode.ZERO_OR_ONE:
            labels[cat.name] = [filtered[0]] if filtered else []
        elif cat.mode == CategoryMode.ZERO_OR_MORE:
            labels[cat.name] = filtered

    logger.debug("qwen_vl: %s -> %s", Path(image_path).name, labels)
    return labels


def _qwen_vl_chat(
    config: QwenVLConfig,
    image_path: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 512,
) -> str:
    """Send a chat request to a Qwen2.5-VL endpoint and return the text response.

    Shared helper for classification, description, and VQA calls.
    Handles base64 encoding, think-tag stripping, and error handling.
    """
    img_data, media_type = _encode_image(image_path)

    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{img_data}",
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "max_tokens": max_tokens,
    }

    logger.debug("qwen_vl: POST %s (model=%s) image=%s",
                 config.endpoint, config.model, Path(image_path).name)

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
        logger.debug("qwen_vl: HTTP %d: %s", e.code, error_body)
        raise RuntimeError(
            f"Qwen VL request failed (HTTP {e.code}): {error_body[:200]}"
        ) from e

    raw_content = result["choices"][0]["message"]["content"]
    content = re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()

    logger.debug("qwen_vl: raw response: %s", raw_content[:300])
    logger.debug("qwen_vl: cleaned response: %s", content[:300])
    return content


def run_qwen_vl_describe(
    config: QwenVLConfig,
    image_path: str,
) -> str:
    """Ask Qwen2.5-VL to describe an image in free-form text.

    Used by suggest workflows as a replacement for BLIP open-ended captioning.
    Returns a plain-text description.
    """
    system_prompt = (
        "You are an image analysis assistant. Describe what you see in the image "
        "in detail. Focus on the subject, setting, objects, actions, and style. "
        "Be concise but thorough."
    )
    user_text = "Describe this image."

    logger.debug("qwen_vl_describe: %s", Path(image_path).name)
    return _qwen_vl_chat(config, image_path, system_prompt, user_text, max_tokens=256)


def run_qwen_vl_vqa(
    config: QwenVLConfig,
    image_path: str,
    question: str,
    max_tokens: int = 100,
) -> str:
    """Ask Qwen2.5-VL a specific question about an image.

    Used by suggest workflows as a replacement for BLIP VQA.
    Returns a plain-text answer.
    """
    system_prompt = (
        "You are an image analysis assistant. Answer the question about the image. "
        "Be concise and direct."
    )

    logger.debug("qwen_vl_vqa: %s question=%r", Path(image_path).name, question)
    return _qwen_vl_chat(config, image_path, system_prompt, question, max_tokens=max_tokens)


def _test_post(url: str, body: dict, timeout: float = 10.0) -> tuple[bool, str]:
    """POST JSON to a URL and return (ok, message).

    Success means the server returned a 2xx with parseable JSON containing
    the expected chat-completions structure.
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            # Verify it looks like a chat completion response
            content = result["choices"][0]["message"]["content"]
            preview = content.strip()[:60].replace("\n", " ")
            return True, f"OK — \"{preview}\""
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")[:200]
        return False, f"FAIL (HTTP {e.code}: {error_body})"
    except urllib.error.URLError as e:
        return False, f"FAIL ({e.reason})"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return False, f"FAIL (unexpected response: {e})"
    except Exception as e:
        return False, f"FAIL ({e})"


def _test_form_post(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """POST a tiny 1x1 PNG to a BLIP/CLIP-style form endpoint.

    These endpoints expect multipart/form-data with a 'file' field.
    We send a minimal image to verify the server accepts requests.
    Any JSON response counts as success. A structured error from the
    model (e.g. 'cannot process') still proves the endpoint is live.
    """
    # 1x1 red PNG (67 bytes)
    import struct
    import zlib
    def _make_tiny_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        # IHDR
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        # IDAT
        raw_row = b"\x00\xff\x00\x00"  # filter=none, R=255, G=0, B=0
        compressed = zlib.compress(raw_row)
        idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
        idat = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)
        # IEND
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return sig + ihdr + idat + iend

    png_bytes = _make_tiny_png()
    boundary = "----labelman-test-boundary"
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"test.png\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + png_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
            # BLIP returns {"caption": "..."}, CLIP returns {"scores": {...}}
            if "caption" in result or "answer" in result:
                text = result.get("caption") or result.get("answer", "")
                preview = text.strip()[:60]
                return True, f"OK — \"{preview}\""
            elif "scores" in result:
                n = len(result["scores"])
                return True, f"OK — {n} score(s) returned"
            else:
                return True, f"OK — {json.dumps(result)[:60]}"
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")[:200]
        return False, f"FAIL (HTTP {e.code}: {error_body})"
    except urllib.error.URLError as e:
        return False, f"FAIL ({e.reason})"
    except Exception as e:
        return False, f"FAIL ({e})"


def test_llm_endpoint(config: LLMConfig, timeout: float = 10.0) -> tuple[bool, str]:
    """Test an LLM endpoint by sending a trivial chat completion."""
    body: dict = {
        "messages": [
            {"role": "user", "content": "Say OK."},
        ],
        "max_tokens": 4,
    }
    if config.model:
        body["model"] = config.model
    return _test_post(config.endpoint, body, timeout=timeout)


def test_qwen_vl_endpoint(config: QwenVLConfig, timeout: float = 10.0) -> tuple[bool, str]:
    """Test a Qwen VL endpoint by sending a text-only chat completion."""
    body = {
        "model": config.model,
        "messages": [
            {"role": "user", "content": "Say OK."},
        ],
        "max_tokens": 4,
    }
    return _test_post(config.endpoint, body, timeout=timeout)


def test_blip_endpoint(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Test a BLIP endpoint by POSTing a tiny test image."""
    return _test_form_post(url, timeout=timeout)


def test_clip_endpoint(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Test a CLIP endpoint by POSTing a tiny test image."""
    return _test_form_post(url, timeout=timeout)
