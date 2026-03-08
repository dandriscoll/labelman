"""Tests for the web-based labeling interface."""

import json
from http.client import HTTPConnection
from pathlib import Path

import pytest

from labelman.label import load_manual_sidecar
from labelman.web import ImageIndex, load_detected_sidecar, start_server_background, write_manual_sidecar


# --- Fixtures ---

@pytest.fixture
def dataset(tmp_path):
    """Create a dataset directory with dummy images and some sidecars."""
    for i in range(5):
        (tmp_path / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8dummy")
    # Non-image files should be ignored
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "config.yaml").write_text("not an image")
    # Add manual labels to one image
    (tmp_path / "img_001.labels.txt").write_text("label one, label two")
    return tmp_path


@pytest.fixture
def index(dataset):
    return ImageIndex(dataset)


@pytest.fixture
def server(dataset):
    """Start a test server and return (connection, base_url, dataset_path)."""
    srv, port = start_server_background(dataset)
    conn = HTTPConnection("127.0.0.1", port)
    yield conn, dataset
    srv.shutdown()
    conn.close()


# --- ImageIndex tests ---

def test_index_finds_images(index):
    assert index.total == 5


def test_index_ignores_non_images(index):
    items, total = index.list_images()
    all_names = [i["name"] for i in items]
    assert "readme.txt" not in all_names
    assert "config.yaml" not in all_names


def test_index_pagination(index):
    items, total = index.list_images(page=1, per_page=2)
    assert len(items) == 2
    assert total == 5

    items2, _ = index.list_images(page=2, per_page=2)
    assert len(items2) == 2

    items3, _ = index.list_images(page=3, per_page=2)
    assert len(items3) == 1


def test_index_pagination_last_page(index):
    items, total = index.list_images(page=3, per_page=2)
    assert len(items) == 1
    assert total == 5


def test_index_has_labels(index):
    items, _ = index.list_images()
    labeled = [i for i in items if i["has_labels"]]
    assert len(labeled) == 1
    assert labeled[0]["name"] == "img_001.jpg"
    assert labeled[0]["label_count"] == 2


def test_index_resolve_image(index, dataset):
    path = index.resolve_image("img_000.jpg")
    assert path is not None
    assert path.name == "img_000.jpg"


def test_index_resolve_traversal_rejected(index):
    assert index.resolve_image("../etc/passwd") is None
    assert index.resolve_image("../../secret.jpg") is None
    assert index.resolve_image(".hidden") is None


def test_index_resolve_nonexistent(index):
    assert index.resolve_image("nope.jpg") is None


def test_index_resolve_non_image(index):
    assert index.resolve_image("readme.txt") is None


def test_index_get_labels(index):
    labels = index.get_labels("img_001.jpg")
    assert labels == ["label one", "label two"]


def test_index_get_labels_empty(index):
    labels = index.get_labels("img_000.jpg")
    assert labels == []


def test_index_set_labels(index, dataset):
    assert index.set_labels("img_000.jpg", ["new label", "another"])
    sidecar = dataset / "img_000.labels.txt"
    assert sidecar.exists()
    assert sidecar.read_text() == "new label, another"


def test_index_set_labels_nonexistent(index):
    assert not index.set_labels("nope.jpg", ["label"])


def test_index_refresh(index, dataset):
    (dataset / "extra.png").write_bytes(b"\x89PNGdummy")
    assert index.total == 5
    index.refresh()
    assert index.total == 6


# --- write_manual_sidecar ---

def test_write_manual_sidecar(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    sidecar = write_manual_sidecar(img, ["label a", "label b"])
    assert sidecar.name == "photo.labels.txt"
    assert sidecar.read_text() == "label a, label b"


def test_write_manual_sidecar_empty(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    # First create a sidecar
    write_manual_sidecar(img, ["something"])
    # Then clear it
    write_manual_sidecar(img, [])
    sidecar = tmp_path / "photo.labels.txt"
    assert sidecar.read_text() == ""


def test_roundtrip_with_load_manual_sidecar(tmp_path):
    """Labels written by web module can be read by label.load_manual_sidecar."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    write_manual_sidecar(img, ["tail number n123ab", "red stripe livery"])
    loaded = load_manual_sidecar(str(img))
    assert loaded == ["tail number n123ab", "red stripe livery"]


# --- HTTP API tests ---

def _get_json(conn, path):
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    return resp.status, json.loads(body)


def _put_json(conn, path, data):
    body = json.dumps(data).encode()
    conn.request("PUT", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def _post_json(conn, path, data):
    body = json.dumps(data).encode()
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read())


def test_get_root_returns_html(server):
    conn, _ = server
    conn.request("GET", "/")
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert b"<!DOCTYPE html>" in body
    assert "text/html" in resp.getheader("Content-Type")


def test_get_images_list(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/images")
    assert status == 200
    assert data["total"] == 5
    assert len(data["images"]) == 5
    assert data["page"] == 1


def test_get_images_pagination(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/images?page=1&per_page=2")
    assert status == 200
    assert len(data["images"]) == 2
    assert data["total"] == 5
    assert data["pages"] == 3


def test_get_image_thumb(server):
    conn, _ = server
    conn.request("GET", "/api/images/img_000.jpg/thumb")
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "image/jpeg"
    assert body == b"\xff\xd8dummy"


def test_get_image_thumb_missing(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/images/nonexistent.jpg/thumb")
    assert status == 404


def test_get_labels_empty(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/images/img_000.jpg/labels")
    assert status == 200
    assert data["manual_labels"] == []


def test_get_labels_existing(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/images/img_001.jpg/labels")
    assert status == 200
    assert data["manual_labels"] == ["label one", "label two"]


def test_put_labels_creates_sidecar(server):
    conn, dataset = server
    status, data = _put_json(conn, "/api/images/img_002.jpg/labels", {
        "labels": ["new label", "another label"]
    })
    assert status == 200
    assert data["manual_labels"] == ["new label", "another label"]
    # Verify sidecar was created on disk
    sidecar = dataset / "img_002.labels.txt"
    assert sidecar.exists()
    assert "new label" in sidecar.read_text()


def test_put_labels_overwrites(server):
    conn, dataset = server
    # First write
    _put_json(conn, "/api/images/img_003.jpg/labels", {"labels": ["first"]})
    # Overwrite
    status, data = _put_json(conn, "/api/images/img_003.jpg/labels", {"labels": ["second", "third"]})
    assert status == 200
    assert data["manual_labels"] == ["second", "third"]
    sidecar = dataset / "img_003.labels.txt"
    content = sidecar.read_text()
    assert "first" not in content
    assert "second" in content


def test_put_labels_empty(server):
    conn, dataset = server
    # Create labels first
    _put_json(conn, "/api/images/img_003.jpg/labels", {"labels": ["something"]})
    # Clear them
    status, data = _put_json(conn, "/api/images/img_003.jpg/labels", {"labels": []})
    assert status == 200
    assert data["manual_labels"] == []


def test_put_labels_nonexistent(server):
    conn, _ = server
    status, data = _put_json(conn, "/api/images/nope.jpg/labels", {"labels": ["x"]})
    assert status == 404


def test_bulk_add_labels(server):
    conn, dataset = server
    status, data = _post_json(conn, "/api/bulk/labels", {
        "images": ["img_000.jpg", "img_002.jpg"],
        "add": ["bulk tag"],
    })
    assert status == 200
    assert len(data["updated"]) == 2
    for item in data["updated"]:
        assert "bulk tag" in item["manual_labels"]


def test_bulk_remove_labels(server):
    conn, _ = server
    status, data = _post_json(conn, "/api/bulk/labels", {
        "images": ["img_001.jpg"],
        "remove": ["label one"],
    })
    assert status == 200
    assert data["updated"][0]["manual_labels"] == ["label two"]


def test_bulk_add_and_remove(server):
    conn, _ = server
    # First add a label
    _post_json(conn, "/api/bulk/labels", {
        "images": ["img_000.jpg"],
        "add": ["keep this", "remove this"],
    })
    # Then add and remove in one call
    status, data = _post_json(conn, "/api/bulk/labels", {
        "images": ["img_000.jpg"],
        "add": ["new one"],
        "remove": ["remove this"],
    })
    assert status == 200
    labels = data["updated"][0]["manual_labels"]
    assert "keep this" in labels
    assert "new one" in labels
    assert "remove this" not in labels


def test_labels_persist_after_write(server):
    """Labels written via API persist and can be read back via load_manual_sidecar."""
    conn, dataset = server
    _put_json(conn, "/api/images/img_004.jpg/labels", {
        "labels": ["persisted label"]
    })
    loaded = load_manual_sidecar(str(dataset / "img_004.jpg"))
    assert loaded == ["persisted label"]


# --- Delete API ---

def test_delete_image(server):
    conn, dataset = server
    # Add labels first so sidecars exist
    _put_json(conn, "/api/images/img_002.jpg/labels", {"labels": ["tag"]})
    assert (dataset / "img_002.jpg").exists()
    assert (dataset / "img_002.labels.txt").exists()
    # Delete
    conn.request("DELETE", "/api/images/img_002.jpg")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert data["deleted"] == "img_002.jpg"
    assert data["total"] == 4
    assert not (dataset / "img_002.jpg").exists()
    assert not (dataset / "img_002.labels.txt").exists()


def test_delete_image_with_detected_sidecar(server):
    conn, dataset = server
    (dataset / "img_003.txt").write_text("detected stuff")
    conn.request("DELETE", "/api/images/img_003.jpg")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert not (dataset / "img_003.jpg").exists()
    assert not (dataset / "img_003.txt").exists()


def test_delete_image_nonexistent(server):
    conn, _ = server
    conn.request("DELETE", "/api/images/nope.jpg")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 404


# --- CLI integration ---

# --- Taxonomy API ---

def test_api_taxonomy_no_config(server):
    conn, _ = server
    status, data = _get_json(conn, "/api/taxonomy")
    assert status == 200
    assert data["categories"] == []


def test_api_taxonomy_with_config(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")
    config = tmp_path / "labelman.yaml"
    config.write_text("""\
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
    srv, port = start_server_background(tmp_path)
    conn = HTTPConnection("127.0.0.1", port)
    try:
        status, data = _get_json(conn, "/api/taxonomy")
        assert status == 200
        assert len(data["categories"]) == 2
        assert data["categories"][0]["name"] == "subject"
        assert data["categories"][0]["mode"] == "exactly-one"
        assert data["categories"][0]["terms"] == ["person", "animal"]
        assert data["categories"][1]["name"] == "mood"
        assert data["categories"][1]["terms"] == ["calm", "energetic"]
    finally:
        srv.shutdown()
        conn.close()


def test_index_taxonomy_loaded(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")
    config = tmp_path / "labelman.yaml"
    config.write_text("""\
defaults:
  threshold: 0.3
global_terms:
  - aircraft
categories:
  - name: count
    mode: exactly-one
    terms:
      - term: single
      - term: group
""")
    index = ImageIndex(tmp_path)
    tax = index.get_taxonomy()
    assert tax is not None
    assert len(tax["categories"]) == 1
    assert tax["global_terms"] == ["aircraft"]


def test_index_taxonomy_missing(dataset):
    index = ImageIndex(dataset)
    assert index.get_taxonomy() is None


def test_cli_ui_parser():
    from labelman.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["ui", "--images", "/tmp/imgs", "--port", "9999"])
    assert args.command == "ui"
    assert args.images == "/tmp/imgs"
    assert args.port == 9999
    assert args.host == "0.0.0.0"


# --- Detected labels ---

def test_load_detected_sidecar(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    sidecar = tmp_path / "photo.txt"
    sidecar.write_text("aircraft, mooney m20, single, calm")
    labels = load_detected_sidecar(img)
    assert labels == ["aircraft", "mooney m20", "single", "calm"]


def test_load_detected_sidecar_missing(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    assert load_detected_sidecar(img) == []


def test_load_detected_sidecar_empty(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "photo.txt").write_text("")
    assert load_detected_sidecar(img) == []


def test_index_get_detected_labels(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "img.txt").write_text("aircraft, single")
    index = ImageIndex(tmp_path)
    assert index.get_detected_labels("img.jpg") == ["aircraft", "single"]


def test_api_labels_includes_detected(tmp_path):
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")
    (tmp_path / "img.txt").write_text("aircraft, single, calm")
    (tmp_path / "img.labels.txt").write_text("manual tag")
    srv, port = start_server_background(tmp_path)
    conn = HTTPConnection("127.0.0.1", port)
    try:
        status, data = _get_json(conn, "/api/images/img.jpg/labels")
        assert status == 200
        assert data["manual_labels"] == ["manual tag"]
        assert data["detected_labels"] == ["aircraft", "single", "calm"]
    finally:
        srv.shutdown()
        conn.close()


def test_suppress_label_with_minus(tmp_path):
    """Writing -term to manual sidecar is valid and round-trips."""
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8")
    write_manual_sidecar(img, ["-calm", "custom tag"])
    loaded = load_manual_sidecar(str(img))
    assert loaded == ["-calm", "custom tag"]


def test_cli_ui_defaults():
    from labelman.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["ui"])
    assert args.port == 7933
    assert args.host == "0.0.0.0"
