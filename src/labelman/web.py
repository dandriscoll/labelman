"""Web-based manual labeling interface for labelman."""

from __future__ import annotations

import json
import math
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .label import load_manual_sidecar
from .schema import ParseError, parse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}

DEFAULT_PER_PAGE = 50
MAX_REQUEST_BODY = 1_048_576  # 1 MB


class ImageIndex:
    """Scans a directory for images and provides paginated access."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self._images: list[str] = []
        self._taxonomy: dict | None = None
        self.refresh()

    def refresh(self) -> None:
        self._images = sorted(
            p.name for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self._taxonomy = self._load_taxonomy()

    def _load_taxonomy(self) -> dict | None:
        """Load labelman.yaml from the dataset directory if present."""
        config_path = self.directory / "labelman.yaml"
        if not config_path.is_file():
            return None
        try:
            term_list = parse(config_path)
        except ParseError:
            return None
        categories = []
        for cat in term_list.categories:
            categories.append({
                "name": cat.name,
                "mode": cat.mode.value,
                "terms": [t.term for t in cat.terms],
            })
        result = {"categories": categories}
        if term_list.global_terms:
            result["global_terms"] = term_list.global_terms
        return result

    def get_taxonomy(self) -> dict | None:
        return self._taxonomy

    @property
    def total(self) -> int:
        return len(self._images)

    def list_images(self, page: int = 1, per_page: int = DEFAULT_PER_PAGE) -> tuple[list[dict], int]:
        per_page = max(1, min(per_page, 500))
        page = max(1, page)
        start = (page - 1) * per_page
        end = start + per_page
        names = self._images[start:end]

        items = []
        for name in names:
            path = self.directory / name
            manual = load_manual_sidecar(str(path))
            detected = load_detected_sidecar(path)
            total_labels = len(manual) + len(detected)
            items.append({
                "name": name,
                "has_labels": total_labels > 0,
                "label_count": total_labels,
            })
        return items, self.total

    def resolve_image(self, name: str) -> Path | None:
        """Resolve an image name to a safe path within the directory."""
        if "/" in name or "\\" in name or name.startswith("."):
            return None
        path = (self.directory / name).resolve()
        if not str(path).startswith(str(self.directory)):
            return None
        if not path.is_file():
            return None
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            return None
        return path

    def get_labels(self, name: str) -> list[str] | None:
        path = self.resolve_image(name)
        if path is None:
            return None
        return load_manual_sidecar(str(path))

    def get_detected_labels(self, name: str) -> list[str] | None:
        path = self.resolve_image(name)
        if path is None:
            return None
        return load_detected_sidecar(path)

    def set_labels(self, name: str, labels: list[str]) -> bool:
        path = self.resolve_image(name)
        if path is None:
            return False
        write_manual_sidecar(path, labels)
        return True


def write_manual_sidecar(image_path: Path, labels: list[str]) -> Path:
    """Write manual labels to a .labels.txt sidecar file.

    Format matches what load_manual_sidecar expects: comma-separated.
    """
    sidecar = image_path.with_suffix(".labels.txt")
    if labels:
        sidecar.write_text(", ".join(labels))
    else:
        if sidecar.exists():
            sidecar.write_text("")
    return sidecar


def load_detected_sidecar(image_path: Path) -> list[str]:
    """Load detected labels from a .txt sidecar (comma-separated caption format).

    These are the output of 'labelman label' — the final assembled caption.
    Returns an empty list if no sidecar exists.
    """
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return []
    text = sidecar.read_text().strip()
    if not text:
        return []
    return [label.strip() for label in text.split(",") if label.strip()]


class LabelmanHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the labeling web interface."""

    index: ImageIndex

    def log_message(self, format, *args):
        # Suppress default request logging
        pass

    def _send_json(self, data: dict | list, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _read_body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_REQUEST_BODY:
            self._send_error(413, "Request body too large")
            return None
        if length == 0:
            self._send_error(400, "Empty request body")
            return None
        return self.rfile.read(length)

    def _parse_json_body(self) -> dict | None:
        body = self._read_body()
        if body is None:
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return None
        if not isinstance(data, dict):
            self._send_error(400, "Expected JSON object")
            return None
        return data

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/images":
            self._handle_list_images(qs)
        elif path.startswith("/api/images/") and path.endswith("/thumb"):
            name = unquote(path[len("/api/images/"):-len("/thumb")])
            self._handle_get_thumb(name)
        elif path.startswith("/api/images/") and path.endswith("/labels"):
            name = unquote(path[len("/api/images/"):-len("/labels")])
            self._handle_get_labels(name)
        elif path == "/api/taxonomy":
            taxonomy = self.index.get_taxonomy()
            self._send_json(taxonomy if taxonomy else {"categories": []})
        else:
            self._send_error(404, "Not found")

    def do_PUT(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/images/") and path.endswith("/labels"):
            name = unquote(path[len("/api/images/"):-len("/labels")])
            self._handle_put_labels(name)
        elif path.startswith("/api/images/") and path.endswith("/detected"):
            name = unquote(path[len("/api/images/"):-len("/detected")])
            self._handle_put_detected(name)
        else:
            self._send_error(404, "Not found")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path.startswith("/api/images/"):
            name = unquote(path[len("/api/images/"):])
            self._handle_delete_image(name)
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/bulk/labels":
            self._handle_bulk_labels()
        elif path == "/api/refresh":
            self.index.refresh()
            self._send_json({"total": self.index.total})
        else:
            self._send_error(404, "Not found")

    def _serve_html(self) -> None:
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_list_images(self, qs: dict) -> None:
        page = int(qs.get("page", ["1"])[0])
        per_page = int(qs.get("per_page", [str(DEFAULT_PER_PAGE)])[0])
        items, total = self.index.list_images(page, per_page)
        pages = math.ceil(total / max(1, per_page))
        self._send_json({
            "images": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
        })

    def _handle_get_thumb(self, name: str) -> None:
        path = self.index.resolve_image(name)
        if path is None:
            self._send_error(404, "Image not found")
            return
        mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _handle_get_labels(self, name: str) -> None:
        labels = self.index.get_labels(name)
        if labels is None:
            self._send_error(404, "Image not found")
            return
        detected = self.index.get_detected_labels(name) or []
        self._send_json({
            "name": name,
            "manual_labels": labels,
            "detected_labels": detected,
        })

    def _handle_put_labels(self, name: str) -> None:
        data = self._parse_json_body()
        if data is None:
            return
        labels = data.get("labels")
        if not isinstance(labels, list):
            self._send_error(400, "'labels' must be a list")
            return
        labels = [str(l).strip() for l in labels if str(l).strip()]
        if not self.index.set_labels(name, labels):
            self._send_error(404, "Image not found")
            return
        self._send_json({"name": name, "manual_labels": labels})

    def _handle_delete_image(self, name: str) -> None:
        path = self.index.resolve_image(name)
        if path is None:
            self._send_error(404, "Image not found")
            return
        # Remove image and associated sidecars
        manual_sidecar = path.with_suffix(".labels.txt")
        detected_sidecar = path.with_suffix(".txt")
        path.unlink()
        if manual_sidecar.is_file():
            manual_sidecar.unlink()
        if detected_sidecar.is_file():
            detected_sidecar.unlink()
        self.index.refresh()
        self._send_json({"deleted": name, "total": self.index.total})

    def _handle_put_detected(self, name: str) -> None:
        data = self._parse_json_body()
        if data is None:
            return
        text = data.get("text", "")
        if not isinstance(text, str):
            self._send_error(400, "'text' must be a string")
            return
        path = self.index.resolve_image(name)
        if path is None:
            self._send_error(404, "Image not found")
            return
        sidecar = path.with_suffix(".txt")
        sidecar.write_text(text)
        self._send_json({"name": name, "text": text})

    def _handle_bulk_labels(self) -> None:
        data = self._parse_json_body()
        if data is None:
            return
        images = data.get("images", [])
        add = data.get("add", [])
        remove = data.get("remove", [])

        if not isinstance(images, list) or not images:
            self._send_error(400, "'images' must be a non-empty list")
            return

        results = []
        for name in images:
            current = self.index.get_labels(str(name))
            if current is None:
                continue
            updated = list(current)
            for label in remove:
                label = str(label).strip()
                if label in updated:
                    updated.remove(label)
            for label in add:
                label = str(label).strip()
                if label and label not in updated:
                    updated.append(label)
            self.index.set_labels(str(name), updated)
            results.append({"name": str(name), "manual_labels": updated})

        self._send_json({"updated": results})


def _get_display_host(host: str) -> str:
    """Get a usable hostname for display in the URL.

    When binding to 0.0.0.0, resolve the machine's FQDN so the printed
    URL is clickable from other machines.
    """
    import socket
    if host in ("0.0.0.0", "::"):
        try:
            fqdn = socket.getfqdn()
            # getfqdn can return 'localhost' or empty on misconfigured systems
            if fqdn and fqdn != "localhost":
                return fqdn
        except OSError:
            pass
        return socket.gethostname()
    return host


def serve(directory: Path, host: str = "0.0.0.0", port: int = 0) -> None:
    """Start the labeling web interface."""
    index = ImageIndex(directory)

    handler_class = type(
        "BoundHandler",
        (LabelmanHandler,),
        {"index": index},
    )

    server = HTTPServer((host, port), handler_class)
    actual_port = server.server_address[1]
    display_host = _get_display_host(host)
    print(f"Labeling UI: http://{display_host}:{actual_port}")
    print(f"Dataset: {directory} ({index.total} images)")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
        server.shutdown()


def start_server_background(directory: Path, host: str = "127.0.0.1", port: int = 0) -> tuple[HTTPServer, int]:
    """Start the server in a background thread (for testing).

    Returns (server, port).
    """
    index = ImageIndex(directory)
    handler_class = type(
        "BoundHandler",
        (LabelmanHandler,),
        {"index": index},
    )
    server = HTTPServer((host, port), handler_class)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


# --- Embedded HTML/CSS/JS Frontend ---

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>labelman UI</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; overflow: hidden; }
.app { display: flex; height: 100vh; }
.sidebar { width: 320px; min-width: 320px; background: #16213e; display: flex; flex-direction: column; border-right: 1px solid #333; }
.sidebar-header { padding: 12px; background: #0f3460; border-bottom: 1px solid #333; }
.sidebar-header h1 { font-size: 16px; font-weight: 600; color: #e94560; }
.sidebar-header .stats { font-size: 12px; color: #888; margin-top: 4px; }
.controls { padding: 8px 12px; border-bottom: 1px solid #333; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.controls button, .controls select { background: #0f3460; color: #e0e0e0; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 12px; cursor: pointer; }
.controls button:hover { background: #1a5276; }
.controls button.active { background: #e94560; border-color: #e94560; }
.image-list { flex: 1; overflow-y: auto; padding: 4px; }
.image-list.grid-view { display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 4px; align-content: start; }
.image-item { display: flex; align-items: center; padding: 6px 8px; border-radius: 6px; cursor: pointer; gap: 8px; margin-bottom: 2px; border: 2px solid transparent; user-select: none; }
.image-item:hover { background: #1a3a5c; }
.image-item.focused { border-color: #e94560; background: #1a3a5c; }
.image-item.selected { background: #2a1a3e; border-color: #9b59b6; }
.image-item .thumb { width: 40px; height: 40px; object-fit: cover; border-radius: 4px; background: #333; flex-shrink: 0; }
.image-item .info { flex: 1; min-width: 0; }
.image-item .name { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.image-item .label-count { font-size: 11px; color: #888; }
.image-item .label-count.has-labels { color: #4caf50; }
.grid-view .image-item { flex-direction: column; padding: 4px; margin-bottom: 0; gap: 2px; }
.grid-view .image-item .thumb { width: 100%; height: 70px; border-radius: 4px; }
.grid-view .image-item .info { width: 100%; }
.grid-view .image-item .name { font-size: 10px; text-align: center; }
.grid-view .image-item .label-count { font-size: 9px; text-align: center; }
.pagination { padding: 8px 12px; border-top: 1px solid #333; display: flex; justify-content: space-between; align-items: center; font-size: 12px; }
.pagination button { background: #0f3460; color: #e0e0e0; border: 1px solid #444; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
.pagination button:disabled { opacity: 0.3; cursor: not-allowed; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.main-header { padding: 8px 16px; background: #16213e; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; min-height: 44px; }
.main-header .title { font-size: 14px; font-weight: 600; }
.bulk-bar { padding: 8px 16px; background: #2a1a3e; border-bottom: 1px solid #533; display: flex; gap: 8px; align-items: center; }
.bulk-bar input { background: #1a1a2e; color: #e0e0e0; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 13px; flex: 1; }
.bulk-bar button { background: #0f3460; color: #e0e0e0; border: 1px solid #444; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.bulk-bar button.danger { background: #8b0000; }
.main-content { flex: 1; display: flex; overflow: hidden; }
.preview { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #111; }
.preview img { max-width: 100%; max-height: 100%; object-fit: contain; }
.preview .empty { color: #555; font-size: 14px; }
.label-panel { width: 300px; min-width: 300px; background: #16213e; border-left: 1px solid #333; display: flex; flex-direction: column; overflow: hidden; }
.label-panel h3 { padding: 10px 12px; font-size: 13px; border-bottom: 1px solid #333; font-weight: 600; }
.label-section { padding: 8px 12px; flex: 1; overflow-y: auto; }
.label-section h4 { font-size: 11px; text-transform: uppercase; color: #888; margin-bottom: 6px; letter-spacing: 0.5px; }
.label-row { display: flex; align-items: center; padding: 3px 0; border-bottom: 1px solid #222; gap: 4px; }
.label-row .label-text { flex: 1; font-size: 12px; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.label-row.manual .label-text { color: #4caf50; }
.label-row.detected .label-text { color: #5dade2; }
.label-row.promoted .label-text { color: #4caf50; opacity: 0.6; }
.label-row.suppressed .label-text { color: #e94560; text-decoration: line-through; }
.label-row .actions { display: flex; gap: 2px; flex-shrink: 0; }
.label-btn { cursor: pointer; font-weight: bold; font-size: 14px; line-height: 1; padding: 2px 5px; border: none; border-radius: 3px; background: none; }
.label-btn.remove { color: #e94560; }
.label-btn.remove:hover { color: #ff6b6b; background: rgba(233,69,96,0.1); }
.label-btn.promote { color: #4caf50; }
.label-btn.promote:hover { color: #66bb66; background: rgba(76,175,80,0.1); }
.label-btn.suppress { color: #e94560; }
.label-btn.suppress:hover { color: #ff6b6b; background: rgba(233,69,96,0.1); }
.label-btn.undo { color: #888; }
.label-btn.undo:hover { color: #ccc; background: rgba(255,255,255,0.05); }
.cat-section { margin-bottom: 10px; }
.cat-header { font-size: 11px; text-transform: uppercase; color: #888; letter-spacing: 0.5px; margin-bottom: 4px; display: flex; align-items: center; gap: 6px; }
.cat-mode { font-size: 9px; color: #666; font-weight: normal; text-transform: none; }
.term-grid { display: flex; flex-wrap: wrap; gap: 3px; }
.term-btn { font-size: 12px; padding: 3px 10px; border-radius: 4px; border: 1px solid #444; background: #1a1a2e; color: #aaa; cursor: pointer; transition: all 0.1s; display: inline-flex; align-items: center; gap: 4px; }
.term-btn:hover { border-color: #666; color: #ddd; }
.term-btn.detected { background: #1a2a3e; border-color: #2980b9; color: #5dade2; }
.term-btn.active { background: #1e2a1e; border-color: #4caf50; color: #6c6; }
.term-btn.partial { background: #2a2a1e; border-color: #b8860b; color: #daa520; }
.term-btn.suppressed { background: #2a1a1e; border-color: #e94560; color: #e94560; text-decoration: line-through; opacity: 0.7; }
.term-btn .term-remove { font-size: 11px; color: #e94560; margin-left: 2px; }
.add-label { display: flex; gap: 4px; margin-top: 8px; }
.add-label input { flex: 1; background: #1a1a2e; color: #e0e0e0; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
.add-label button { background: #0f3460; color: #e0e0e0; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.divider { height: 1px; background: #333; margin: 12px 0; }
.kbd { background: #333; border: 1px solid #555; border-radius: 3px; padding: 1px 5px; font-size: 10px; font-family: monospace; color: #aaa; }
.help-bar { padding: 6px 12px; border-top: 1px solid #333; font-size: 11px; color: #666; display: flex; gap: 12px; flex-wrap: wrap; }
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="sidebar-header">
      <h1>labelman</h1>
      <div class="stats" id="stats">Loading...</div>
    </div>
    <div class="controls">
      <button id="btn-view-toggle">Grid</button>
      <button id="btn-refresh">Refresh</button>
      <select id="per-page">
        <option value="25">25</option>
        <option value="50" selected>50</option>
        <option value="100">100</option>
        <option value="200">200</option>
      </select>
    </div>
    <div class="image-list" id="image-list"></div>
    <div class="pagination">
      <button id="btn-prev" disabled>&larr; Prev</button>
      <span id="page-info">-</span>
      <button id="btn-next" disabled>Next &rarr;</button>
    </div>
  </div>
  <div class="main">
    <div class="main-header">
      <span class="title" id="current-name">No image selected</span>
      <span id="save-status" style="font-size:12px;color:#4caf50"></span>
    </div>
    <div class="main-content">
      <div class="preview" id="preview">
        <span class="empty">Select an image to preview</span>
      </div>
      <div class="label-panel" id="label-panel">
        <h3>Labels</h3>
        <div class="label-section" id="label-section">
          <p style="color:#666;font-size:12px;padding:8px 0">Select an image to view labels</p>
        </div>
        <div id="add-section" style="display:none;padding:8px 12px;border-top:1px solid #333">
          <div class="add-label">
            <input id="add-input" placeholder="New label..." />
            <button id="btn-add">Add</button>
          </div>
        </div>
        <div id="raw-section" style="display:none;padding:8px 12px;border-top:1px solid #333">
          <h4 style="font-size:11px;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:4px">Manual (.labels.txt)</h4>
          <textarea id="raw-manual" rows="3" style="width:100%;background:#1a1a2e;color:#4caf50;border:1px solid #444;border-radius:4px;font-size:11px;font-family:monospace;padding:4px;resize:vertical"></textarea>
          <h4 style="font-size:11px;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin:6px 0 4px">Detected (.txt)</h4>
          <textarea id="raw-detected" rows="2" style="width:100%;background:#1a1a2e;color:#5dade2;border:1px solid #444;border-radius:4px;font-size:11px;font-family:monospace;padding:4px;resize:vertical"></textarea>
        </div>
        <div id="delete-section" style="display:none;padding:8px 12px;border-top:1px solid #333">
          <button id="btn-delete" style="width:100%;background:#8b0000;color:#e0e0e0;border:1px solid #e94560;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:600">Delete Image</button>
        </div>
        <div class="help-bar">
          <span><span class="kbd">&larr;</span><span class="kbd">&rarr;</span> navigate</span>
          <span><span class="kbd">Space</span> select</span>
          <span><span class="kbd">Enter</span> edit</span>
          <span><span class="kbd">Del</span> delete image</span>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
(function() {
  let images = [];
  let total = 0;
  let page = 1;
  let perPage = 50;
  let pages = 1;
  let focusIdx = -1;
  let selectedSet = new Set();
  let currentLabels = [];
  let detectedLabels = [];
  let taxonomy = null;
  let lastShiftIdx = -1;
  let viewMode = 'list'; // 'list' or 'grid'
  let multiLabelData = {}; // {name: {manual: [], detected: []}}

  const $list = document.getElementById('image-list');
  const $stats = document.getElementById('stats');
  const $pageInfo = document.getElementById('page-info');
  const $preview = document.getElementById('preview');
  const $labelSection = document.getElementById('label-section');
  const $currentName = document.getElementById('current-name');
  const $addInput = document.getElementById('add-input');
  const $addSection = document.getElementById('add-section');
  const $rawSection = document.getElementById('raw-section');
  const $rawManual = document.getElementById('raw-manual');
  const $rawDetected = document.getElementById('raw-detected');
  const $saveStatus = document.getElementById('save-status');

  async function api(url, opts) {
    const r = await fetch(url, opts);
    return r.json();
  }

  function isMultiSelect() { return viewMode === 'grid' && selectedSet.size > 1; }

  async function loadImages() {
    const data = await api(`/api/images?page=${page}&per_page=${perPage}`);
    images = data.images;
    total = data.total;
    pages = data.pages;
    $stats.textContent = `${total} images`;
    $pageInfo.textContent = `${page} / ${pages}`;
    document.getElementById('btn-prev').disabled = page <= 1;
    document.getElementById('btn-next').disabled = page >= pages;
    renderList();
    if (viewMode === 'list' && images.length > 0 && focusIdx < 0) {
      focusIdx = 0;
      selectImage(0);
    } else if (focusIdx >= images.length) {
      focusIdx = Math.max(0, images.length - 1);
    }
  }

  function renderList() {
    $list.innerHTML = '';
    $list.className = viewMode === 'grid' ? 'image-list grid-view' : 'image-list';
    images.forEach((img, i) => {
      const el = document.createElement('div');
      const isFocused = viewMode === 'list' && i === focusIdx;
      const isSelected = viewMode === 'grid' && selectedSet.has(img.name);
      el.className = 'image-item' + (isFocused ? ' focused' : '') + (isSelected ? ' selected' : '');
      el.innerHTML = `
        <img class="thumb" src="/api/images/${encodeURIComponent(img.name)}/thumb" loading="lazy" />
        <div class="info">
          <div class="name" title="${img.name}">${img.name}</div>
          <div class="label-count ${img.label_count > 0 ? 'has-labels' : ''}">${img.label_count} label${img.label_count !== 1 ? 's' : ''}</div>
        </div>`;
      el.addEventListener('click', (e) => {
        if (viewMode === 'grid') {
          handleGridClick(i, e);
        } else {
          focusIdx = i;
          selectedSet.clear();
          selectImage(i);
          renderList();
        }
      });
      $list.appendChild(el);
    });
  }

  function handleGridClick(idx, e) {
    const name = images[idx].name;
    if (e.shiftKey && lastShiftIdx >= 0) {
      const start = Math.min(lastShiftIdx, idx);
      const end = Math.max(lastShiftIdx, idx);
      for (let i = start; i <= end; i++) {
        selectedSet.add(images[i].name);
      }
    } else if (e.ctrlKey || e.metaKey) {
      if (selectedSet.has(name)) selectedSet.delete(name);
      else selectedSet.add(name);
    } else {
      selectedSet.clear();
      selectedSet.add(name);
    }
    lastShiftIdx = idx;
    focusIdx = idx;
    renderList();
    onSelectionChanged();
  }

  function onSelectionChanged() {
    const names = Array.from(selectedSet);
    if (names.length === 0) {
      $currentName.textContent = 'No image selected';
      $preview.innerHTML = '<span class="empty">Select an image to preview</span>';
      $labelSection.innerHTML = '<p style="color:#666;font-size:12px;padding:8px 0">Select an image to view labels</p>';
      $addSection.style.display = 'none';
      $rawSection.style.display = 'none';
      document.getElementById('delete-section').style.display = 'none';
    } else if (names.length === 1) {
      $currentName.textContent = names[0];
      $preview.innerHTML = `<img src="/api/images/${encodeURIComponent(names[0])}/thumb" />`;
      $addSection.style.display = 'block';
      $rawSection.style.display = 'block';
      document.getElementById('delete-section').style.display = 'block';
      loadLabels(names[0]);
    } else {
      $currentName.textContent = `${names.length} images selected`;
      $preview.innerHTML = `<span class="empty">${names.length} images selected</span>`;
      $addSection.style.display = 'block';
      $rawSection.style.display = 'none';
      document.getElementById('delete-section').style.display = 'none';
      loadMultiLabels(names);
    }
  }

  function selectImage(idx) {
    if (idx < 0 || idx >= images.length) return;
    const img = images[idx];
    $currentName.textContent = img.name;
    $preview.innerHTML = `<img src="/api/images/${encodeURIComponent(img.name)}/thumb" />`;
    $addSection.style.display = 'block';
    $rawSection.style.display = 'block';
    document.getElementById('delete-section').style.display = 'block';
    loadLabels(img.name);
  }

  async function loadLabels(name) {
    const data = await api(`/api/images/${encodeURIComponent(name)}/labels`);
    currentLabels = data.manual_labels || [];
    detectedLabels = data.detected_labels || [];
    renderLabels(name);
  }

  async function loadMultiLabels(names) {
    multiLabelData = {};
    const results = await Promise.all(names.map(n =>
      api(`/api/images/${encodeURIComponent(n)}/labels`).then(d => ({name: n, data: d}))
    ));
    for (const r of results) {
      multiLabelData[r.name] = {
        manual: r.data.manual_labels || [],
        detected: r.data.detected_labels || [],
      };
    }
    renderBulkLabels(names);
  }

  function renderLabels(name) {
    const additive = currentLabels.filter(l => !l.startsWith('-'));
    const suppressions = new Set(currentLabels.filter(l => l.startsWith('-')).map(l => l.slice(1)));
    const taxonomyTerms = new Set();
    let html = '';

    if (taxonomy && taxonomy.categories && taxonomy.categories.length > 0) {
      taxonomy.categories.forEach(cat => {
        const modeLabel = cat.mode === 'exactly-one' ? 'pick one' : cat.mode === 'zero-or-one' ? 'optional, pick one' : 'any';
        html += `<div class="cat-section"><div class="cat-header">${esc(cat.name)} <span class="cat-mode">${modeLabel}</span></div><div class="term-grid">`;
        const isExclusive = cat.mode === 'exactly-one' || cat.mode === 'zero-or-one';
        const hasManualSelection = isExclusive && cat.terms.some(t => additive.includes(t));
        cat.terms.forEach(term => {
          taxonomyTerms.add(term);
          const isManual = additive.includes(term);
          const isDetected = detectedLabels.includes(term);
          const isSuppressed = suppressions.has(term);
          const isImplicitlySuppressed = !isSuppressed && !isManual && hasManualSelection;
          let cls = 'term-btn';
          if (isSuppressed || isImplicitlySuppressed) cls += ' suppressed';
          else if (isManual) cls += ' active';
          else if (isDetected) cls += ' detected';
          html += `<button class="${cls}" data-term="${esc(term)}" data-cat="${esc(cat.name)}" data-mode="${cat.mode}">${esc(term)}</button>`;
        });
        html += '</div></div>';
      });
    }

    const extraManual = additive.filter(l => !taxonomyTerms.has(l));
    if (extraManual.length > 0) {
      html += '<div class="divider"></div><h4>Other Manual Labels</h4>';
      extraManual.forEach(l => {
        const realIdx = currentLabels.indexOf(l);
        html += `<div class="label-row manual">
          <span class="label-text" title="${esc(l)}">${esc(l)}</span>
          <div class="actions"><button class="label-btn remove" data-idx="${realIdx}">&times;</button></div>
        </div>`;
      });
    }

    const extraDetected = detectedLabels.filter(l => !taxonomyTerms.has(l));
    if (extraDetected.length > 0) {
      html += '<div class="divider"></div><h4>Detected (non-taxonomy)</h4>';
      extraDetected.forEach(l => {
        const isPromoted = additive.includes(l);
        const isSuppressed = suppressions.has(l);
        let cls = 'label-row detected';
        if (isPromoted) cls += ' promoted';
        if (isSuppressed) cls += ' suppressed';
        html += `<div class="${cls}"><span class="label-text" title="${esc(l)}">${esc(l)}</span><div class="actions">`;
        if (!isPromoted && !isSuppressed) {
          html += `<button class="label-btn promote" data-label="${esc(l)}" title="Add as manual label">+</button>`;
          html += `<button class="label-btn suppress" data-label="${esc(l)}" title="Suppress this label">&#x2212;</button>`;
        } else {
          const undoLabel = isSuppressed ? '-' + l : l;
          html += `<button class="label-btn undo" data-label="${esc(undoLabel)}" title="Undo">&times;</button>`;
        }
        html += '</div></div>';
      });
    }

    if (!taxonomy || !taxonomy.categories || taxonomy.categories.length === 0) {
      html = '<h4>Manual Labels</h4>';
      if (additive.length === 0) {
        html += '<p style="color:#666;font-size:12px">No manual labels</p>';
      } else {
        additive.forEach(l => {
          const realIdx = currentLabels.indexOf(l);
          html += `<div class="label-row manual">
            <span class="label-text" title="${esc(l)}">${esc(l)}</span>
            <div class="actions"><button class="label-btn remove" data-idx="${realIdx}">&times;</button></div>
          </div>`;
        });
      }
      html += '<div class="divider"></div><h4>Detected Labels</h4>';
      if (detectedLabels.length === 0) {
        html += '<p style="color:#666;font-size:12px">No detected labels</p>';
      } else {
        detectedLabels.forEach(l => {
          const isPromoted = additive.includes(l);
          const isSuppressed = suppressions.has(l);
          let cls = 'label-row detected';
          if (isPromoted) cls += ' promoted';
          if (isSuppressed) cls += ' suppressed';
          html += `<div class="${cls}"><span class="label-text" title="${esc(l)}">${esc(l)}</span><div class="actions">`;
          if (!isPromoted && !isSuppressed) {
            html += `<button class="label-btn promote" data-label="${esc(l)}" title="Add as manual label">+</button>`;
            html += `<button class="label-btn suppress" data-label="${esc(l)}" title="Suppress this label">&#x2212;</button>`;
          } else {
            const undoLabel = isSuppressed ? '-' + l : l;
            html += `<button class="label-btn undo" data-label="${esc(undoLabel)}" title="Undo">&times;</button>`;
          }
          html += '</div></div>';
        });
      }
    }

    $labelSection.innerHTML = html;
    $rawManual.value = currentLabels.join(', ');
    $rawDetected.value = detectedLabels.join(', ');
    bindLabelEvents(name);
  }

  // --- Bulk / multi-select label rendering ---
  function renderBulkLabels(names) {
    const n = names.length;
    // Aggregate: for each term, count how many images have it as manual/detected
    function countTerm(term) {
      let manual = 0, detected = 0, suppressed = 0;
      for (const name of names) {
        const d = multiLabelData[name];
        if (!d) continue;
        const additive = d.manual.filter(l => !l.startsWith('-'));
        const supps = new Set(d.manual.filter(l => l.startsWith('-')).map(l => l.slice(1)));
        if (supps.has(term)) suppressed++;
        else if (additive.includes(term)) manual++;
        else if (d.detected.includes(term)) detected++;
      }
      return {manual, detected, suppressed};
    }
    function countLabel(label) {
      let count = 0;
      for (const name of names) {
        const d = multiLabelData[name];
        if (!d) continue;
        if (d.manual.includes(label)) count++;
      }
      return count;
    }

    const taxonomyTerms = new Set();
    let html = '';

    if (taxonomy && taxonomy.categories && taxonomy.categories.length > 0) {
      taxonomy.categories.forEach(cat => {
        const modeLabel = cat.mode === 'exactly-one' ? 'pick one' : cat.mode === 'zero-or-one' ? 'optional, pick one' : 'any';
        html += `<div class="cat-section"><div class="cat-header">${esc(cat.name)} <span class="cat-mode">${modeLabel}</span></div><div class="term-grid">`;
        cat.terms.forEach(term => {
          taxonomyTerms.add(term);
          const c = countTerm(term);
          let cls = 'term-btn';
          let badge = '';
          if (c.manual === n) cls += ' active';
          else if (c.manual > 0) { cls += ' partial'; badge = ` <span style="font-size:9px;opacity:0.7">${c.manual}/${n}</span>`; }
          else if (c.detected > 0) { cls += ' detected'; badge = c.detected < n ? ` <span style="font-size:9px;opacity:0.7">${c.detected}/${n}</span>` : ''; }
          if (c.suppressed === n) cls = 'term-btn suppressed';
          else if (c.suppressed > 0 && c.manual === 0 && c.detected === 0) { cls = 'term-btn suppressed'; badge = ` <span style="font-size:9px;opacity:0.7">${c.suppressed}/${n}</span>`; }
          html += `<button class="${cls}" data-term="${esc(term)}" data-cat="${esc(cat.name)}" data-mode="${cat.mode}">${esc(term)}${badge}</button>`;
        });
        html += '</div></div>';
      });
    }

    // Collect all extra manual labels across selection
    const allExtraManual = new Set();
    for (const name of names) {
      const d = multiLabelData[name];
      if (!d) continue;
      for (const l of d.manual) {
        if (!l.startsWith('-') && !taxonomyTerms.has(l)) allExtraManual.add(l);
      }
    }
    if (allExtraManual.size > 0) {
      html += '<div class="divider"></div><h4>Other Manual Labels</h4>';
      for (const l of allExtraManual) {
        const c = countLabel(l);
        const badge = c < n ? ` <span style="font-size:9px;color:#888">${c}/${n}</span>` : '';
        html += `<div class="label-row manual">
          <span class="label-text" title="${esc(l)}">${esc(l)}${badge}</span>
          <div class="actions"><button class="label-btn remove" data-bulk-label="${esc(l)}">&times;</button></div>
        </div>`;
      }
    }

    $labelSection.innerHTML = html;
    bindBulkLabelEvents(names);
  }

  function bindBulkLabelEvents(names) {
    $labelSection.querySelectorAll('.term-btn').forEach(el => {
      el.addEventListener('click', async () => {
        const term = el.dataset.term;
        const mode = el.dataset.mode;
        const cat = el.dataset.cat;
        const isAll = el.classList.contains('active');
        $saveStatus.textContent = 'Saving...';
        if (isAll) {
          // Remove from all
          await api('/api/bulk/labels', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({images: names, remove: [term]}),
          });
        } else {
          // Add to all; for exclusive categories, remove siblings first
          const payload = {images: names, add: [term]};
          if (mode === 'exactly-one' || mode === 'zero-or-one') {
            const catDef = taxonomy && taxonomy.categories.find(c => c.name === cat);
            if (catDef) {
              payload.remove = catDef.terms.filter(t => t !== term);
            }
          }
          await api('/api/bulk/labels', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
          });
        }
        $saveStatus.textContent = 'Saved';
        setTimeout(() => { $saveStatus.textContent = ''; }, 1500);
        await loadMultiLabels(names);
      });
    });
    $labelSection.querySelectorAll('.label-btn.remove[data-bulk-label]').forEach(el => {
      el.addEventListener('click', async () => {
        const label = el.dataset.bulkLabel;
        $saveStatus.textContent = 'Saving...';
        await api('/api/bulk/labels', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({images: names, remove: [label]}),
        });
        $saveStatus.textContent = 'Saved';
        setTimeout(() => { $saveStatus.textContent = ''; }, 1500);
        await loadMultiLabels(names);
      });
    });
  }

  function clearCategoryExclusives(term, cat, mode) {
    if (mode !== 'exactly-one' && mode !== 'zero-or-one') return;
    const catDef = taxonomy && taxonomy.categories.find(c => c.name === cat);
    if (!catDef) return;
    catDef.terms.forEach(t => {
      if (t === term) return;
      const mIdx = currentLabels.indexOf(t);
      if (mIdx >= 0) currentLabels.splice(mIdx, 1);
    });
  }

  function bindLabelEvents(name) {
    $labelSection.querySelectorAll('.term-btn').forEach(el => {
      el.addEventListener('click', () => {
        const term = el.dataset.term;
        const mode = el.dataset.mode;
        const cat = el.dataset.cat;
        const isManual = el.classList.contains('active');
        const isSuppressed = el.classList.contains('suppressed');

        if (isSuppressed) {
          const idx = currentLabels.indexOf('-' + term);
          if (idx >= 0) currentLabels.splice(idx, 1);
        } else if (isManual) {
          const idx = currentLabels.indexOf(term);
          if (idx >= 0) currentLabels.splice(idx, 1);
        } else {
          clearCategoryExclusives(term, cat, mode);
          const sIdx = currentLabels.indexOf('-' + term);
          if (sIdx >= 0) currentLabels.splice(sIdx, 1);
          if (!currentLabels.includes(term)) {
            currentLabels.push(term);
          }
        }
        saveLabels(name);
      });
      el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const term = el.dataset.term;
        const isManual = el.classList.contains('active');
        const isSuppressed = el.classList.contains('suppressed');
        if (isSuppressed) {
          const idx = currentLabels.indexOf('-' + term);
          if (idx >= 0) currentLabels.splice(idx, 1);
        } else {
          if (isManual) {
            const idx = currentLabels.indexOf(term);
            if (idx >= 0) currentLabels.splice(idx, 1);
          }
          if (!currentLabels.includes('-' + term)) {
            currentLabels.push('-' + term);
          }
        }
        saveLabels(name);
      });
    });

    $labelSection.querySelectorAll('.label-btn.remove').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.dataset.idx);
        currentLabels.splice(idx, 1);
        saveLabels(name);
      });
    });
    $labelSection.querySelectorAll('.label-btn.promote').forEach(el => {
      el.addEventListener('click', () => {
        const label = el.dataset.label;
        if (!currentLabels.includes(label)) currentLabels.push(label);
        saveLabels(name);
      });
    });
    $labelSection.querySelectorAll('.label-btn.suppress').forEach(el => {
      el.addEventListener('click', () => {
        const label = '-' + el.dataset.label;
        if (!currentLabels.includes(label)) currentLabels.push(label);
        saveLabels(name);
      });
    });
    $labelSection.querySelectorAll('.label-btn.undo').forEach(el => {
      el.addEventListener('click', () => {
        const label = el.dataset.label;
        const idx = currentLabels.indexOf(label);
        if (idx >= 0) currentLabels.splice(idx, 1);
        saveLabels(name);
      });
    });
  }

  async function saveLabels(name) {
    $saveStatus.textContent = 'Saving...';
    await api(`/api/images/${encodeURIComponent(name)}/labels`, {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({labels: currentLabels}),
    });
    $saveStatus.textContent = 'Saved';
    setTimeout(() => { $saveStatus.textContent = ''; }, 1500);
    renderLabels(name);
    const img = images.find(i => i.name === name);
    if (img) {
      const totalCount = currentLabels.length + detectedLabels.length;
      img.label_count = totalCount;
      img.has_labels = totalCount > 0;
      renderList();
    }
  }

  async function addLabel() {
    const val = $addInput.value.trim();
    if (!val) return;
    if (isMultiSelect()) {
      // Bulk add
      const names = Array.from(selectedSet);
      $saveStatus.textContent = 'Saving...';
      await api('/api/bulk/labels', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({images: names, add: [val]}),
      });
      $addInput.value = '';
      $saveStatus.textContent = 'Saved';
      setTimeout(() => { $saveStatus.textContent = ''; }, 1500);
      await loadMultiLabels(names);
      return;
    }
    if (focusIdx < 0) return;
    const name = images[focusIdx].name;
    if (!currentLabels.includes(val)) {
      currentLabels.push(val);
    }
    $addInput.value = '';
    await saveLabels(name);
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // Keyboard navigation
  function navNext() {
    if (focusIdx < images.length - 1) {
      focusIdx++;
      if (viewMode === 'list') { selectImage(focusIdx); }
      renderList();
      scrollToFocused();
    } else if (page < pages) {
      page++;
      focusIdx = 0;
      loadImages().then(() => { if (viewMode === 'list') selectImage(0); });
    }
  }
  function navPrev() {
    if (focusIdx > 0) {
      focusIdx--;
      if (viewMode === 'list') { selectImage(focusIdx); }
      renderList();
      scrollToFocused();
    } else if (page > 1) {
      page--;
      loadImages().then(() => {
        focusIdx = images.length - 1;
        if (viewMode === 'list') { selectImage(focusIdx); }
        renderList();
      });
    }
  }
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && (e.key === 'ArrowRight' || e.key === 'ArrowDown')) {
      e.preventDefault();
      navNext();
      return;
    }
    if (e.ctrlKey && (e.key === 'ArrowLeft' || e.key === 'ArrowUp')) {
      e.preventDefault();
      navPrev();
      return;
    }
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      if (e.key === 'Enter' && e.target.id === 'add-input') {
        e.preventDefault();
        addLabel();
      }
      if (e.key === 'Escape') {
        e.target.blur();
      }
      return;
    }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'j') {
      e.preventDefault();
      navNext();
    }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp' || e.key === 'k') {
      e.preventDefault();
      navPrev();
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      $addInput.focus();
    }
    if (e.key === 'Escape') {
      selectedSet.clear();
      renderList();
      if (viewMode === 'grid') onSelectionChanged();
    }
    if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (viewMode === 'grid') {
        images.forEach(img => selectedSet.add(img.name));
        renderList();
        onSelectionChanged();
      }
    }
    if (e.key === 'Delete' && focusIdx >= 0 && !isMultiSelect()) {
      e.preventDefault();
      const name = images[focusIdx].name;
      if (confirm(`Delete ${name} and its label files?`)) {
        deleteCurrentImage();
      }
    }
  });

  function scrollToFocused() {
    const el = $list.querySelector('.focused') || $list.querySelector('.selected');
    if (el) el.scrollIntoView({block: 'nearest'});
  }

  // View toggle
  document.getElementById('btn-view-toggle').addEventListener('click', () => {
    viewMode = viewMode === 'list' ? 'grid' : 'list';
    document.getElementById('btn-view-toggle').textContent = viewMode === 'list' ? 'Grid' : 'List';
    selectedSet.clear();
    if (viewMode === 'list') {
      if (focusIdx < 0 && images.length > 0) focusIdx = 0;
      if (focusIdx >= 0) selectImage(focusIdx);
    } else {
      onSelectionChanged();
    }
    renderList();
  });

  // Event listeners
  document.getElementById('btn-prev').addEventListener('click', () => { if (page > 1) { page--; loadImages(); } });
  document.getElementById('btn-next').addEventListener('click', () => { if (page < pages) { page++; loadImages(); } });
  document.getElementById('btn-refresh').addEventListener('click', async () => {
    await api('/api/refresh', {method: 'POST'});
    await loadImages();
  });
  document.getElementById('per-page').addEventListener('change', (e) => {
    perPage = parseInt(e.target.value);
    page = 1;
    loadImages();
  });
  document.getElementById('btn-add').addEventListener('click', addLabel);

  // Auto-save raw text fields with debounce (single-select only)
  let rawManualTimer = null;
  let rawDetectedTimer = null;
  $rawManual.addEventListener('input', () => {
    clearTimeout(rawManualTimer);
    rawManualTimer = setTimeout(() => {
      if (focusIdx < 0 || isMultiSelect()) return;
      const name = images[focusIdx].name;
      const text = $rawManual.value.trim();
      currentLabels = text ? text.split(',').map(s => s.trim()).filter(Boolean) : [];
      saveLabels(name);
    }, 600);
  });
  $rawDetected.addEventListener('input', () => {
    clearTimeout(rawDetectedTimer);
    rawDetectedTimer = setTimeout(async () => {
      if (focusIdx < 0 || isMultiSelect()) return;
      const name = images[focusIdx].name;
      const text = $rawDetected.value.trim();
      $saveStatus.textContent = 'Saving...';
      await fetch(`/api/images/${encodeURIComponent(name)}/detected`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text}),
      });
      detectedLabels = text ? text.split(',').map(s => s.trim()).filter(Boolean) : [];
      $saveStatus.textContent = 'Saved';
      setTimeout(() => { $saveStatus.textContent = ''; }, 1500);
    }, 600);
  });

  // Delete image
  async function deleteCurrentImage() {
    if (focusIdx < 0) return;
    const name = images[focusIdx].name;
    await fetch(`/api/images/${encodeURIComponent(name)}`, {method: 'DELETE'});
    selectedSet.delete(name);
    await loadImages();
    if (focusIdx >= images.length) focusIdx = Math.max(0, images.length - 1);
    if (images.length > 0) {
      if (viewMode === 'list') { selectImage(focusIdx); }
      renderList();
    } else {
      $currentName.textContent = 'No image selected';
      $preview.innerHTML = '<span class="empty">No images remaining</span>';
      $labelSection.innerHTML = '';
      $addSection.style.display = 'none';
      $rawSection.style.display = 'none';
      document.getElementById('delete-section').style.display = 'none';
    }
  }
  document.getElementById('btn-delete').addEventListener('click', deleteCurrentImage);

  // Initial load
  api('/api/taxonomy').then(data => { taxonomy = data; }).catch(() => {});
  loadImages();
})();
</script>
</body>
</html>
"""
