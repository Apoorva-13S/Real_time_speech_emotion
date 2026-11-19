#!/usr/bin/env python3
"""Tiny web demo: upload a clip, get emotion probabilities.

Standard library only (http.server) so it runs anywhere without Flask, and it
degrades honestly: if no trained checkpoint is present it says so instead of
silently serving a different model's predictions.

    python app/server.py --model runs/local/best --port 8000
    python app/server.py --self-test          # no model needed, for CI

Endpoints
    GET  /            the page
    GET  /healthz     {"status":"ok","model_loaded":bool,...}
    POST /predict     multipart form field `file` -> JSON probabilities
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_UPLOAD_BYTES = 12 * 1024 * 1024          # 12 MB, a 4 s clip is ~250 KB
ALLOWED_EXT = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}

STATE: dict = {"predictor": None, "error": None, "model": None}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(model_dir: str | None, allow_fallback: bool) -> None:
    from ser.predict import EmotionPredictor

    try:
        STATE["predictor"] = EmotionPredictor(model_dir, verbose=True)
        STATE["model"] = str(STATE["predictor"].model_dir)
        STATE["fallback"] = bool(STATE["predictor"].fallback)
        if STATE["fallback"] and not allow_fallback:
            raise FileNotFoundError(
                "no trained checkpoint found and fallback disabled")
        STATE["error"] = None
    except Exception as exc:                              # noqa: BLE001
        STATE["predictor"] = None
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"model not loaded: {STATE['error']}")


# --------------------------------------------------------------------------- #
# Multipart parsing (enough for one small file field)
# --------------------------------------------------------------------------- #
def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes] | None:
    if "boundary=" not in content_type:
        return None
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()
    for part in body.split(sep):
        if b"Content-Disposition" not in part:
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if b'name="file"' not in head and b'name="audio"' not in head:
            continue
        filename = ""
        for token in head.decode("utf-8", "replace").split(";"):
            token = token.strip()
            if token.startswith("filename="):
                filename = token.split("=", 1)[1].strip().strip('"')
        return filename, data.rstrip(b"\r\n")
    return None


def wav_duration(blob: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(blob)) as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:                                    # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "SERDemo/1.0"

    def log_message(self, fmt: str, *args) -> None:      # quieter logs
        sys.stderr.write("  " + fmt % args + "\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, indent=2).encode(), "application/json")

    # ---------------------------------------------------------------- #
    def do_GET(self) -> None:                            # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/healthz", "/api/health"):
            self._json(200, {
                "status": "ok",
                "model_loaded": STATE["predictor"] is not None,
                "model": STATE["model"],
                "fallback_model": STATE.get("fallback", False),
                "error": STATE["error"],
            })
            return
        if path in ("/", "/index.html"):
            html = (ROOT / "app" / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return
        if path == "/api/labels":
            pred = STATE["predictor"]
            self._json(200, {"labels": pred.classes if pred else []})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:                           # noqa: N802
        if self.path.split("?")[0] not in ("/predict", "/api/predict"):
            self._json(404, {"error": "not found"})
            return
        if STATE["predictor"] is None:
            self._json(503, {
                "error": "no model loaded",
                "detail": STATE["error"],
                "hint": "Run training first (see process.txt) and start the "
                        "server with --model runs/local/best",
            })
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json(400, {"error": "empty request"})
                return
            if length > MAX_UPLOAD_BYTES:
                self._json(413, {"error": f"file too large (limit "
                                         f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB)"})
                return
            body = self.rfile.read(length)
            parsed = parse_multipart(body, self.headers.get("Content-Type", ""))
            if not parsed:
                self._json(400, {"error": "expected multipart form field 'file'"})
                return
            filename, blob = parsed
            ext = Path(filename).suffix.lower()
            if ext and ext not in ALLOWED_EXT:
                self._json(415, {"error": f"unsupported file type {ext}",
                                 "allowed": sorted(ALLOWED_EXT)})
                return
            if len(blob) < 512:
                self._json(400, {"error": "file looks empty or truncated"})
                return

            tmp = Path(tempfile.mkstemp(suffix=ext or ".wav")[1])
            tmp.write_bytes(blob)
            try:
                preds = STATE["predictor"].predict_file(tmp, top_k=None)
                dur = wav_duration(blob)
            finally:
                tmp.unlink(missing_ok=True)

            self._json(200, {
                "file": filename,
                "duration_s": round(dur, 2) if dur else None,
                "model": STATE["model"],
                "fallback_model": STATE.get("fallback", False),
                "top": preds[0],
                "predictions": preds,
                "note": ("This is an estimate from acted speech and is not a "
                         "measurement of anyone's real emotional state."),
            })
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- #
def self_test() -> int:
    """Verify the HTTP plumbing without a model or a microphone."""
    import threading
    import urllib.request

    body = io.BytesIO()
    with wave.open(body, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    blob = body.getvalue()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        health = json.loads(urllib.request.urlopen(base + "/healthz").read())
        assert health["status"] == "ok", health
        page = urllib.request.urlopen(base + "/").read()
        assert b"<html" in page.lower(), "index.html not served"

        boundary = "----selftestboundary"
        payload = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + blob + \
            f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            base + "/predict", data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            resp = json.loads(urllib.request.urlopen(req).read())
            code = 200
        except urllib.error.HTTPError as err:            # 503 when no model
            resp = json.loads(err.read())
            code = err.code
        assert code in (200, 503), (code, resp)
        if code == 503:
            assert "no model loaded" in resp["error"], resp
        else:
            assert "predictions" in resp and resp["predictions"], resp
        print(f"self-test OK (predict returned HTTP {code}, "
              f"model_loaded={health['model_loaded']})")
        return 0
    finally:
        srv.shutdown()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Speech emotion web demo")
    ap.add_argument("--model", default=os.environ.get("SER_MODEL",
                                                      "runs/local/best"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--no-fallback", action="store_true",
                    help="refuse to start with a public model if no checkpoint")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    load_model(args.model, allow_fallback=not args.no_fallback)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}  "
          f"(model_loaded={STATE['predictor'] is not None})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
