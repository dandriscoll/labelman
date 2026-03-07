"""Tests for the integration resolution module."""

import json
from pathlib import Path

import pytest

from labelman.integrations import get_descriptor, get_descriptor_path, resolve_script
from labelman.schema import IntegrationConfig


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
