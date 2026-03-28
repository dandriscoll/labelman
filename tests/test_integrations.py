"""Tests for the integration resolution module."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path

import pytest

from labelman.integrations import get_descriptor, get_descriptor_path, resolve_script, run_llm
from labelman.schema import IntegrationConfig, LLMConfig


def test_get_descriptor_blip():
    desc = get_descriptor("blip")
    assert desc["name"] == "labelman-blip"
    assert any(i["id"] == "endpoint" for i in desc["inputs"])
    assert any(i["id"] == "images" for i in desc["inputs"])


def test_get_descriptor_clip():
    desc = get_descriptor("clip")
    assert desc["name"] == "labelman-clip"
    assert any(i["id"] == "endpoint" for i in desc["inputs"])
    assert any(i["id"] == "terms" for i in desc["inputs"])
    assert any(i["id"] == "images" for i in desc["inputs"])


def test_get_descriptor_path_exists():
    for tool in ("blip", "clip"):
        path = get_descriptor_path(tool)
        assert path.is_file()
        data = json.loads(path.read_text())
        assert "name" in data


def test_get_descriptor_unknown():
    with pytest.raises(ValueError, match="Unknown integration"):
        get_descriptor("dall-e")


def test_descriptor_is_valid_json():
    for tool in ("blip", "clip"):
        path = get_descriptor_path(tool)
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        assert "inputs" in data
        assert "output-files" in data


def test_resolve_script_with_endpoint():
    config = IntegrationConfig(endpoint="http://localhost:8080/caption")
    script, endpoint = resolve_script("blip", config)
    assert script.endswith("blip.sh")
    assert endpoint == "http://localhost:8080/caption"
    assert Path(script).is_file()


def test_resolve_script_with_custom_script():
    config = IntegrationConfig(script="/usr/local/bin/my-blip.sh")
    script, endpoint = resolve_script("blip", config)
    assert script == "/usr/local/bin/my-blip.sh"
    assert endpoint is None


def test_resolve_script_custom_ignores_endpoint():
    config = IntegrationConfig(
        endpoint="http://localhost:8080/caption",
        script="/usr/local/bin/my-blip.sh",
    )
    script, endpoint = resolve_script("blip", config)
    assert script == "/usr/local/bin/my-blip.sh"
    assert endpoint is None


def test_resolve_script_none_config():
    with pytest.raises(ValueError, match="not configured"):
        resolve_script("clip", None)


def test_builtin_scripts_are_executable():
    for tool in ("blip", "clip"):
        config = IntegrationConfig(endpoint="http://example.com")
        script, _ = resolve_script(tool, config)
        path = Path(script)
        assert path.is_file()
        import os
        assert os.access(path, os.X_OK)


class _MockLLMHandler(BaseHTTPRequestHandler):
    """Minimal handler that echoes back a fixed chat completion response."""
    response_content = "natural"

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        # Store the request for assertions
        self.server.last_request = body
        reply = {
            "choices": [{"message": {"content": self.response_content}}],
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # suppress logging


class TestRunLLM:
    def _start_server(self):
        server = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        return server

    def test_sends_request_and_returns_content(self):
        server = self._start_server()
        port = server.server_address[1]
        config = LLMConfig(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions", model="test-model")

        result = run_llm(config, "You are a classifier.", "sunlight through window")
        assert result == "natural"
        assert server.last_request["model"] == "test-model"
        assert server.last_request["messages"][0]["role"] == "system"
        assert server.last_request["messages"][1]["role"] == "user"
        assert server.last_request["messages"][1]["content"] == "sunlight through window"
        server.server_close()

    def test_no_model_omits_model_key(self):
        server = self._start_server()
        port = server.server_address[1]
        config = LLMConfig(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions")

        run_llm(config, "system", "user")
        assert "model" not in server.last_request
        server.server_close()

    def test_strips_think_tags(self):
        """Reasoning models like qwen3 wrap output in <think>...</think> tags."""
        _MockLLMHandler.response_content = (
            "<think>\nThe image shows sunlight, which is natural.\n</think>\nnatural"
        )
        server = self._start_server()
        port = server.server_address[1]
        config = LLMConfig(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions")

        result = run_llm(config, "Classify this.", "bright sunshine")
        assert result == "natural"
        server.server_close()
        _MockLLMHandler.response_content = "natural"  # reset
