"""Tests for the Qwen2.5-VL integration."""

import base64
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path

import pytest

from labelman.integrations import (
    _build_category_spec,
    _build_qwen_vl_system_prompt,
    _build_qwen_vl_user_prompt,
    _encode_image,
    run_qwen_vl,
)
from labelman.label import apply_vlm_labels, assemble_final_labels
from labelman.schema import QwenVLConfig, parse


TAXONOMY = """\
defaults:
  threshold: 0.3
integrations:
  qwen_vl:
    endpoint: http://127.0.0.1:{port}/v1/chat/completions
    model: Qwen2.5-VL-7B-Instruct
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: setting
    mode: zero-or-one
    terms:
      - term: indoor
      - term: outdoor
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
      - term: tense
"""


def _parse_with_port(port):
    return parse(TAXONOMY.format(port=port))


class _MockQwenVLHandler(BaseHTTPRequestHandler):
    """Mock handler that returns configurable VLM classification responses."""
    response_labels = {"count": "single", "setting": "indoor", "mood": ["calm"]}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.last_request = body
        reply = {
            "choices": [{
                "message": {
                    "content": json.dumps(self.response_labels),
                },
            }],
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


def _start_server():
    server = HTTPServer(("127.0.0.1", 0), _MockQwenVLHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    return server


# --- Schema parsing ---

class TestQwenVLConfig:
    def test_parse_qwen_vl_config(self):
        tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  qwen_vl:
    endpoint: http://localhost:8082/v1/chat/completions
    model: Qwen2.5-VL-7B-Instruct
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
""")
        assert tl.integrations.qwen_vl is not None
        assert tl.integrations.qwen_vl.endpoint == "http://localhost:8082/v1/chat/completions"
        assert tl.integrations.qwen_vl.model == "Qwen2.5-VL-7B-Instruct"

    def test_parse_qwen_vl_default_model(self):
        tl = parse("""\
defaults:
  threshold: 0.3
integrations:
  qwen_vl:
    endpoint: http://localhost:8082/v1/chat/completions
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
""")
        assert tl.integrations.qwen_vl.model == "Qwen2.5-VL-7B-Instruct"

    def test_parse_qwen_vl_missing_endpoint(self):
        from labelman.schema import ParseError
        with pytest.raises(ParseError, match="endpoint"):
            parse("""\
defaults:
  threshold: 0.3
integrations:
  qwen_vl:
    model: test
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
""")

    def test_parse_without_qwen_vl(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
""")
        assert tl.integrations.qwen_vl is None


# --- Prompt building ---

class TestBuildPrompt:
    def test_system_prompt_contains_allowed_terms(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
""")
        prompt = _build_qwen_vl_system_prompt(tl)
        assert "count" in prompt
        assert "mood" in prompt
        assert "single" in prompt
        assert "group" in prompt
        assert "calm" in prompt
        assert "energetic" in prompt

    def test_system_prompt_embeds_valid_json_taxonomy(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
""")
        prompt = _build_qwen_vl_system_prompt(tl)
        # Extract the JSON block from the prompt (outermost { ... })
        lines = prompt.strip().split("\n")
        json_lines = []
        depth = 0
        for line in lines:
            if "{" in line and depth == 0:
                depth = 1
                json_lines.append(line)
                continue
            if depth > 0:
                json_lines.append(line)
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    break
        taxonomy = json.loads("\n".join(json_lines))
        assert taxonomy["count"]["terms"] == ["single", "group"]
        assert taxonomy["mood"]["terms"] == ["calm"]
        # Each category should include a rule
        assert "rule" in taxonomy["count"]
        assert "rule" in taxonomy["mood"]

    def test_category_spec_returns_term_list(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: tense
        threshold: 0.5
""")
        spec = _build_category_spec(tl.categories[0])
        assert spec == ["calm", "tense"]

    def test_system_prompt_no_mode_or_threshold_leakage(self):
        """The prompt should not contain mode names or threshold values."""
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
  - name: setting
    mode: zero-or-one
    threshold: 0.4
    terms:
      - term: indoor
      - term: outdoor
""")
        prompt = _build_qwen_vl_system_prompt(tl)
        assert "exactly-one" not in prompt
        assert "zero-or-one" not in prompt
        assert "zero-or-more" not in prompt
        assert "0.3" not in prompt
        assert "0.4" not in prompt
        assert "threshold" not in prompt.lower()

    def test_user_prompt_is_short(self):
        prompt = _build_qwen_vl_user_prompt()
        assert "JSON" in prompt
        assert len(prompt) < 100


# --- Image encoding ---

class TestEncodeImage:
    def test_encode_jpeg(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-data")
        data, media_type = _encode_image(str(img))
        assert media_type == "image/jpeg"
        assert base64.b64decode(data) == b"\xff\xd8\xff\xe0fake-jpeg-data"

    def test_encode_png(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-data")
        data, media_type = _encode_image(str(img))
        assert media_type == "image/png"


# --- VLM label application ---

class TestApplyVlmLabels:
    def test_creates_image_labels(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
""")
        vlm_labels = {"subject": ["person"]}
        result = apply_vlm_labels(tl, "img.jpg", vlm_labels)
        assert result.image == "img.jpg"
        assert result.labels == {"subject": ["person"]}
        assert result.all_scores == {}

    def test_assemble_with_vlm_labels(self):
        tl = parse("""\
defaults:
  threshold: 0.3
global_terms:
  - base_tag
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
""")
        vlm_labels = {"subject": ["person"]}
        result = assemble_final_labels(
            tl, "img.jpg", scores={}, vlm_labels=vlm_labels,
        )
        assert "base_tag" in result.final_labels
        assert "person" in result.final_labels

    def test_assemble_vlm_with_manual_suppression(self):
        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
      - term: animal
  - name: mood
    mode: zero-or-more
    terms:
      - term: calm
      - term: energetic
""")
        vlm_labels = {"subject": ["person"], "mood": ["calm", "energetic"]}
        result = assemble_final_labels(
            tl, "img.jpg", scores={},
            manual_labels=["-calm"],
            vlm_labels=vlm_labels,
        )
        assert "calm" not in result.final_labels
        assert "energetic" in result.final_labels
        assert "person" in result.final_labels


# --- End-to-end with mock server ---

class TestRunQwenVL:
    def test_basic_classification(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-data")

        _MockQwenVLHandler.response_labels = {
            "count": "single",
            "setting": "indoor",
            "mood": ["calm"],
        }
        server = _start_server()
        port = server.server_address[1]
        tl = _parse_with_port(port)

        labels = run_qwen_vl(tl, str(img))
        assert labels["count"] == ["single"]
        assert labels["setting"] == ["indoor"]
        assert labels["mood"] == ["calm"]

        # Verify request structure
        req = server.last_request
        assert req["model"] == "Qwen2.5-VL-7B-Instruct"
        # System message with allowed terms
        assert req["messages"][0]["role"] == "system"
        system_content = req["messages"][0]["content"]
        assert "count" in system_content
        assert "single" in system_content
        assert "group" in system_content
        # User message with image + text
        assert req["messages"][1]["role"] == "user"
        user_content = req["messages"][1]["content"]
        assert any(c["type"] == "image_url" for c in user_content)
        assert any(c["type"] == "text" for c in user_content)
        server.server_close()

    def test_filters_invalid_terms(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        _MockQwenVLHandler.response_labels = {
            "count": "triple",  # not a valid term
            "setting": "outdoor",
            "mood": ["calm", "invalid_mood"],
        }
        server = _start_server()
        port = server.server_address[1]
        tl = _parse_with_port(port)

        labels = run_qwen_vl(tl, str(img))
        # exactly-one: invalid term filtered, falls back to first term
        assert labels["count"] == ["single"]
        assert labels["setting"] == ["outdoor"]
        # invalid_mood filtered out
        assert labels["mood"] == ["calm"]
        server.server_close()

    def test_exactly_one_picks_first_when_multiple(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        _MockQwenVLHandler.response_labels = {
            "count": ["single", "group"],  # VLM returned list for exactly-one
            "setting": None,
            "mood": [],
        }
        server = _start_server()
        port = server.server_address[1]
        tl = _parse_with_port(port)

        labels = run_qwen_vl(tl, str(img))
        assert labels["count"] == ["single"]  # first valid term
        assert labels["setting"] == []
        assert labels["mood"] == []
        server.server_close()

    def test_handles_think_tags(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        _MockQwenVLHandler.response_labels = "raw_override"
        # Override to return think-wrapped JSON
        original_do_post = _MockQwenVLHandler.do_POST

        def think_post(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            reply = {
                "choices": [{
                    "message": {
                        "content": '<think>\nLet me analyze...\n</think>\n{"count": "group", "setting": "outdoor", "mood": ["energetic"]}',
                    },
                }],
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        _MockQwenVLHandler.do_POST = think_post
        server = _start_server()
        port = server.server_address[1]
        tl = _parse_with_port(port)

        labels = run_qwen_vl(tl, str(img))
        assert labels["count"] == ["group"]
        assert labels["setting"] == ["outdoor"]
        assert labels["mood"] == ["energetic"]

        _MockQwenVLHandler.do_POST = original_do_post
        server.server_close()

    def test_handles_markdown_code_fences(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        original_do_post = _MockQwenVLHandler.do_POST

        def fence_post(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            reply = {
                "choices": [{
                    "message": {
                        "content": '```json\n{"count": "single", "setting": "indoor", "mood": ["calm"]}\n```',
                    },
                }],
            }
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        _MockQwenVLHandler.do_POST = fence_post
        server = _start_server()
        port = server.server_address[1]
        tl = _parse_with_port(port)

        labels = run_qwen_vl(tl, str(img))
        assert labels["count"] == ["single"]

        _MockQwenVLHandler.do_POST = original_do_post
        server.server_close()

    def test_not_configured_raises(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake")

        tl = parse("""\
defaults:
  threshold: 0.3
categories:
  - name: subject
    mode: exactly-one
    terms:
      - term: person
""")
        from labelman.errors import IntegrationError
        with pytest.raises(IntegrationError, match="not configured"):
            run_qwen_vl(tl, str(img))


# --- Image downscaling ---

class TestEncodeImage:
    def test_downscale_large_image(self, tmp_path):
        from PIL import Image
        import base64 as _b64, io as _io
        from labelman.integrations import _encode_image

        img_path = tmp_path / "big.jpg"
        Image.new("RGB", (4000, 3000), color=(128, 128, 128)).save(img_path, "JPEG", quality=95)
        orig_pixels = 4000 * 3000

        data, media = _encode_image(str(img_path), max_pixels=1_000_000)
        assert media == "image/jpeg"
        decoded = _b64.b64decode(data)
        with Image.open(_io.BytesIO(decoded)) as out:
            w, h = out.size
            assert w * h <= 1_000_000
            # Aspect ratio preserved within rounding tolerance
            assert abs((w / h) - (4000 / 3000)) < 0.02

    def test_small_image_not_resized(self, tmp_path):
        from PIL import Image
        import base64 as _b64, io as _io
        from labelman.integrations import _encode_image

        img_path = tmp_path / "small.jpg"
        Image.new("RGB", (100, 100), color=(64, 64, 64)).save(img_path, "JPEG")

        data, media = _encode_image(str(img_path), max_pixels=1_000_000)
        decoded = _b64.b64decode(data)
        with Image.open(_io.BytesIO(decoded)) as out:
            assert out.size == (100, 100)

    def test_max_pixels_zero_disables(self, tmp_path):
        from PIL import Image
        import base64 as _b64, io as _io
        from labelman.integrations import _encode_image

        img_path = tmp_path / "big.jpg"
        Image.new("RGB", (3000, 2000), color=(0, 0, 0)).save(img_path, "JPEG")

        data, media = _encode_image(str(img_path), max_pixels=0)
        decoded = _b64.b64decode(data)
        with Image.open(_io.BytesIO(decoded)) as out:
            assert out.size == (3000, 2000)

    def test_unreadable_falls_through(self, tmp_path):
        from labelman.integrations import _encode_image
        import base64 as _b64

        img_path = tmp_path / "bogus.jpg"
        img_path.write_bytes(b"not an image")

        data, media = _encode_image(str(img_path), max_pixels=1_000_000)
        assert _b64.b64decode(data) == b"not an image"
        assert media == "image/jpeg"
