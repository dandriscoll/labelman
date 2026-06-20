"""Tests for server-side thumbnail generation and the /full endpoint (issue #1).

The list/grid views request downscaled thumbnails so the browser does not have
to decode full-resolution originals (the cause of the multi-second UI hang). The
preview pane uses /full to keep full resolution for zoom/crop.
"""

import io
from http.client import HTTPConnection

import pytest

from labelman import web
from labelman.web import DEFAULT_THUMB_SIZE, MAX_THUMB_SIZE, start_server_background


@pytest.fixture
def real_dataset(tmp_path):
    """A dataset containing one genuine, large image plus a non-image file."""
    from PIL import Image

    big = Image.new("RGB", (2000, 1500), (200, 100, 50))
    big.save(tmp_path / "big.jpg", quality=90)
    # A file that is not a decodable image — must fall back to original bytes.
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8notreallyanimage")
    return tmp_path


@pytest.fixture
def real_server(real_dataset):
    web._thumb_cache.clear()
    srv, port = start_server_background(real_dataset)
    conn = HTTPConnection("127.0.0.1", port)
    yield conn, real_dataset
    srv.shutdown()
    conn.close()


def _fetch(conn, path):
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read()
    return resp.status, dict(resp.getheaders()), body


def test_thumb_is_downscaled(real_server):
    conn, dataset = real_server
    status, headers, body = _fetch(conn, "/api/images/big.jpg/thumb")
    assert status == 200
    assert headers["Content-Type"] == "image/jpeg"
    from PIL import Image

    thumb = Image.open(io.BytesIO(body))
    # No dimension exceeds the default thumb size.
    assert max(thumb.size) <= DEFAULT_THUMB_SIZE
    # And it is dramatically smaller than the original bytes.
    original = (dataset / "big.jpg").read_bytes()
    assert len(body) < len(original)


def test_thumb_size_param_and_clamp(real_server):
    conn, _ = real_server
    from PIL import Image

    _, _, body = _fetch(conn, "/api/images/big.jpg/thumb?size=128")
    assert max(Image.open(io.BytesIO(body)).size) <= 128
    # Oversized request is clamped to MAX_THUMB_SIZE (image is 2000px wide).
    _, _, body_big = _fetch(conn, "/api/images/big.jpg/thumb?size=99999")
    assert max(Image.open(io.BytesIO(body_big)).size) <= MAX_THUMB_SIZE


def test_thumb_falls_back_for_non_image(real_server):
    conn, _ = real_server
    status, headers, body = _fetch(conn, "/api/images/broken.jpg/thumb")
    assert status == 200
    # Undecodable file is served verbatim (preserves robustness + existing test).
    assert body == b"\xff\xd8notreallyanimage"


def test_thumb_is_cached(real_server):
    conn, _ = real_server
    _, _, first = _fetch(conn, "/api/images/big.jpg/thumb")
    assert len(web._thumb_cache) >= 1
    _, _, second = _fetch(conn, "/api/images/big.jpg/thumb")
    assert first == second


def test_full_returns_original_bytes(real_server):
    conn, dataset = real_server
    from PIL import Image

    status, headers, body = _fetch(conn, "/api/images/big.jpg/full")
    assert status == 200
    assert body == (dataset / "big.jpg").read_bytes()
    # /full is the genuine full-resolution image (preview/zoom/crop accuracy).
    assert max(Image.open(io.BytesIO(body)).size) == 2000


def test_full_missing_is_404(real_server):
    conn, _ = real_server
    conn.request("GET", "/api/images/nope.jpg/full")
    assert conn.getresponse().status == 404
