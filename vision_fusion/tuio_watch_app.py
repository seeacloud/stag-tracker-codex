from __future__ import annotations

import argparse
import json
import math
import mimetypes
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ASSET_DIR = Path(__file__).with_name("watch_app")
TUIO_STALE_SECONDS = 2.0
ANALYTICS_SAVE_INTERVAL = 1.0


@dataclass(slots=True)
class SharedTuioState:
    source: str = "waiting"
    frame: int = -1
    analytics_path: Path | None = None
    objects: dict[int, dict[str, float | int]] = field(default_factory=dict)
    recent_sessions: dict[int, dict[str, float | int]] = field(default_factory=dict)
    analytics: dict[int, dict[str, float | int]] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    last_saved_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_message(self, address: str, args: list[object]) -> None:
        if address != "/tuio/2Dobj" or not args:
            return
        command = args[0]
        with self.lock:
            if command == "source" and len(args) >= 2:
                self.source = str(args[1])
                self.updated_at = time.time()
            elif command == "set" and len(args) >= 11:
                self._set_object(args)
            elif command == "alive":
                alive = {int(value) for value in args[1:] if isinstance(value, int)}
                self.objects = {
                    session_id: obj
                    for session_id, obj in self.objects.items()
                    if session_id in alive
                }
                self.updated_at = time.time()
            elif command == "fseq" and len(args) >= 2:
                self.frame = int(args[1])
                self.updated_at = time.time()

    def _set_object(self, args: list[object]) -> None:
        now = time.time()
        session_id = int(args[1])
        symbol_id = int(args[2])
        previous = self.objects.get(session_id) or self.recent_sessions.get(session_id)
        record = self._analytics_record_locked(symbol_id, now)
        if (
            previous is None
            or int(previous["symbolId"]) != symbol_id
            or now - float(previous["lastSeen"]) > TUIO_STALE_SECONDS
        ):
            record["visits"] = int(record["visits"]) + 1
        elif 0.0 < now - float(previous["lastSeen"]) <= TUIO_STALE_SECONDS:
            record["dwellSeconds"] = float(record["dwellSeconds"]) + now - float(previous["lastSeen"])
        record["lastSeen"] = now

        self.objects[session_id] = {
            "sessionId": session_id,
            "symbolId": symbol_id,
            "x": clamp01(float(args[3])),
            "y": clamp01(float(args[4])),
            "angle": float(args[5]),
            "xVelocity": float(args[6]),
            "yVelocity": float(args[7]),
            "angleVelocity": float(args[8]),
            "motionAccel": float(args[9]),
            "rotationAccel": float(args[10]),
            "lastSeen": now,
        }
        self.recent_sessions[session_id] = {
            "symbolId": symbol_id,
            "lastSeen": now,
        }
        self.updated_at = now
        self._save_analytics_locked()

    def set_demo_objects(self, frame: int, objects: list[dict[str, float | int]]) -> None:
        with self.lock:
            self.source = "demo"
            self.frame = frame
            for obj in objects:
                session_id = int(obj["sessionId"])
                symbol_id = int(obj["symbolId"])
                now = time.time()
                previous = self.objects.get(session_id)
                record = self._analytics_record_locked(symbol_id, now)
                if previous is None or int(previous["symbolId"]) != symbol_id:
                    record["visits"] = int(record["visits"]) + 1
                else:
                    record["dwellSeconds"] = float(record["dwellSeconds"]) + 1 / 30
                record["lastSeen"] = now
                obj["lastSeen"] = now
            self.objects = {int(obj["sessionId"]): obj for obj in objects}
            self.updated_at = time.time()
            self._save_analytics_locked()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            self._prune_stale_locked()
            return {
                "source": self.source,
                "frame": self.frame,
                "updatedAt": self.updated_at,
                "now": time.time(),
                "objects": sorted(
                    self.objects.values(),
                    key=lambda obj: (int(obj["symbolId"]), int(obj["sessionId"])),
                ),
                "analytics": self._analytics_snapshot_locked(),
            }

    def analytics_snapshot(self) -> dict[str, object]:
        with self.lock:
            self._prune_stale_locked()
            return {
                "now": time.time(),
                "analytics": self._analytics_snapshot_locked(),
            }

    def record_interaction(self, payload: dict[str, object]) -> dict[str, object]:
        symbol_id = int(payload.get("symbolId", -1))
        session_id = int(payload.get("sessionId", -1))
        if symbol_id < 0:
            raise ValueError("symbolId is required")
        now = time.time()
        with self.lock:
            record = self._analytics_record_locked(symbol_id, now)
            record["touches"] = int(record["touches"]) + 1
            record["lastInteraction"] = now
            if session_id >= 0:
                record["lastSessionId"] = session_id
            self._save_analytics_locked(force=True)
            return {
                "ok": True,
                "symbolId": symbol_id,
                "analytics": self._analytics_snapshot_locked(),
            }

    def reset_analytics(self) -> dict[str, object]:
        with self.lock:
            self.analytics.clear()
            self.recent_sessions.clear()
            self._save_analytics_locked(force=True)
            return {"ok": True, "analytics": []}

    def load_analytics(self) -> None:
        if self.analytics_path is None or not self.analytics_path.exists():
            return
        try:
            data = json.loads(self.analytics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        records = data.get("analytics", data)
        if not isinstance(records, list):
            return
        with self.lock:
            for raw in records:
                if not isinstance(raw, dict) or "symbolId" not in raw:
                    continue
                symbol_id = int(raw["symbolId"])
                record = self._analytics_record_locked(symbol_id, time.time())
                for key in (
                    "visits",
                    "touches",
                    "dwellSeconds",
                    "firstSeen",
                    "lastSeen",
                    "lastInteraction",
                    "lastSessionId",
                ):
                    if key in raw:
                        record[key] = raw[key]  # type: ignore[assignment]

    def _prune_stale_locked(self) -> None:
        now = time.time()
        self.objects = {
            session_id: obj
            for session_id, obj in self.objects.items()
            if now - float(obj["lastSeen"]) <= TUIO_STALE_SECONDS
        }
        self.recent_sessions = {
            session_id: obj
            for session_id, obj in self.recent_sessions.items()
            if now - float(obj["lastSeen"]) <= TUIO_STALE_SECONDS
        }

    def _analytics_record_locked(self, symbol_id: int, now: float) -> dict[str, float | int]:
        if symbol_id not in self.analytics:
            self.analytics[symbol_id] = {
                "symbolId": symbol_id,
                "visits": 0,
                "touches": 0,
                "dwellSeconds": 0.0,
                "firstSeen": now,
                "lastSeen": now,
                "lastInteraction": 0.0,
                "lastSessionId": -1,
            }
        return self.analytics[symbol_id]

    def _analytics_snapshot_locked(self) -> list[dict[str, float | int]]:
        now = time.time()
        active_counts: dict[int, int] = {}
        for obj in self.objects.values():
            symbol_id = int(obj["symbolId"])
            active_counts[symbol_id] = active_counts.get(symbol_id, 0) + 1

        records: list[dict[str, float | int]] = []
        for symbol_id, record in self.analytics.items():
            active = active_counts.get(symbol_id, 0)
            dwell = float(record["dwellSeconds"])
            visits = int(record["visits"])
            touches = int(record["touches"])
            recency = max(0.0, 30.0 - (now - float(record["lastSeen"]))) / 30.0
            index = dwell * 1.2 + visits * 10.0 + touches * 22.0 + active * 18.0 + recency * 12.0
            output = dict(record)
            output["active"] = active
            output["interestIndex"] = round(index, 1)
            records.append(output)
        return sorted(records, key=lambda item: float(item["interestIndex"]), reverse=True)

    def _save_analytics_locked(self, force: bool = False) -> None:
        if self.analytics_path is None:
            return
        now = time.time()
        if not force and now - self.last_saved_at < ANALYTICS_SAVE_INTERVAL:
            return
        try:
            self.analytics_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "savedAt": now,
                "analytics": self._analytics_snapshot_locked(),
            }
            tmp_path = self.analytics_path.with_suffix(self.analytics_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.analytics_path)
            self.last_saved_at = now
        except OSError as exc:
            print(f"Could not save analytics: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TUIO watch-table display app.")
    parser.add_argument("--tuio-host", default="0.0.0.0", help="UDP address to listen on for TUIO.")
    parser.add_argument("--tuio-port", type=int, default=3333, help="UDP port to listen on for TUIO.")
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP host for the browser app.")
    parser.add_argument("--http-port", type=int, default=8765, help="HTTP port for the browser app.")
    parser.add_argument("--event-hz", type=float, default=30.0, help="SSE update frequency.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic tag objects for preview.")
    parser.add_argument("--analytics-path", default="watch_analytics.json", help="Aggregated analytics JSON path.")
    parser.add_argument("--no-analytics-log", action="store_true", help="Keep analytics in memory only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop = threading.Event()
    analytics_path = None if args.no_analytics_log else Path(args.analytics_path)
    state = SharedTuioState(analytics_path=analytics_path)
    state.load_analytics()

    threads = [
        threading.Thread(
            target=run_tuio_udp_listener,
            args=(args.tuio_host, args.tuio_port, state, stop),
            daemon=True,
        )
    ]
    if args.demo:
        threads.append(threading.Thread(target=run_demo, args=(state, stop), daemon=True))
    for thread in threads:
        thread.start()

    handler = make_handler(state, max(args.event_hz, 1.0))
    server = ThreadingHTTPServer((args.http_host, args.http_port), handler)
    print(f"TUIO watch app: http://{args.http_host}:{args.http_port}/")
    print(f"Listening for /tuio/2Dobj on udp://{args.tuio_host}:{args.tuio_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


def run_tuio_udp_listener(
    host: str,
    port: int,
    state: SharedTuioState,
    stop: threading.Event,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(f"Could not listen for TUIO on {host}:{port}: {exc}")
        sock.close()
        return
    try:
        while not stop.is_set():
            try:
                packet, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            for address, args in parse_osc_packet(packet):
                state.update_message(address, args)
    finally:
        sock.close()


def run_demo(state: SharedTuioState, stop: threading.Event) -> None:
    frame = 0
    start = time.perf_counter()
    while not stop.is_set():
        t = time.perf_counter() - start
        objects: list[dict[str, float | int]] = []
        for index, symbol_id in enumerate((64, 7, 12, 0)):
            phase = t * (0.34 + index * 0.04) + index * math.tau / 4
            radius = 0.16 + index * 0.035
            x = 0.5 + math.cos(phase) * radius
            y = 0.5 + math.sin(phase * 0.88) * radius * 0.72
            objects.append(
                {
                    "sessionId": index + 1,
                    "symbolId": symbol_id,
                    "x": clamp01(x),
                    "y": clamp01(y),
                    "angle": phase + t * 0.4,
                    "xVelocity": -math.sin(phase) * radius,
                    "yVelocity": math.cos(phase * 0.88) * radius,
                    "angleVelocity": 0.4,
                    "motionAccel": 0.0,
                    "rotationAccel": 0.0,
                    "lastSeen": time.time(),
                }
            )
        state.set_demo_objects(frame, objects)
        frame += 1
        time.sleep(1 / 30)


def make_handler(
    state: SharedTuioState,
    event_hz: float,
    asset_dir: Path = ASSET_DIR,
) -> type[BaseHTTPRequestHandler]:
    class WatchAppHandler(BaseHTTPRequestHandler):
        server_version = "TuioWatchApp/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/events":
                self._serve_events()
            elif path == "/api/state":
                self._serve_json(state.snapshot())
            elif path == "/api/analytics":
                self._serve_json(state.analytics_snapshot())
            else:
                self._serve_asset(path)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/interaction":
                self._handle_interaction()
            elif path == "/api/reset-analytics":
                self._serve_json(state.reset_analytics())
            else:
                self.send_error(404)

        def log_message(self, fmt: str, *args: object) -> None:
            if self.path != "/events":
                super().log_message(fmt, *args)

        def _serve_events(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            delay = 1.0 / event_hz
            while True:
                payload = json.dumps(state.snapshot(), separators=(",", ":")).encode("utf-8")
                try:
                    self.wfile.write(b"event: tuio\n")
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
                time.sleep(delay)

        def _serve_json(self, payload: dict[str, object]) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle_interaction(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8") or "{}")
                response = state.record_interaction(payload)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))
                return
            self._serve_json(response)

        def _serve_asset(self, path: str) -> None:
            asset = asset_path(path, asset_dir)
            if asset is None:
                self.send_error(404)
                return
            data = asset.read_bytes()
            # Force correct MIME for web assets. On Windows mimetypes reads the
            # registry, where .js is often text/plain; that makes browsers refuse
            # <script type="module"> (and its imports), white-screening the page.
            forced_types = {
                ".html": "text/html",
                ".js": "text/javascript",
                ".mjs": "text/javascript",
                ".css": "text/css",
            }
            suffix = asset.suffix.lower()
            content_type = (
                forced_types.get(suffix)
                or mimetypes.guess_type(asset.name)[0]
                or "application/octet-stream"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            if asset.suffix.lower() in {".html", ".js", ".css"}:
                self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return WatchAppHandler


def asset_path(path: str, asset_dir: Path = ASSET_DIR) -> Path | None:
    name = "index.html" if path in {"", "/"} else path.lstrip("/")
    candidate = (asset_dir / name).resolve()
    root = asset_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def parse_osc_packet(data: bytes) -> list[tuple[str, list[object]]]:
    if data.startswith(b"#bundle\x00"):
        return parse_osc_bundle(data)
    message = parse_osc_message(data)
    return [message] if message is not None else []


def parse_osc_bundle(data: bytes) -> list[tuple[str, list[object]]]:
    messages: list[tuple[str, list[object]]] = []
    offset = align4(len(b"#bundle\x00")) + 8
    while offset + 4 <= len(data):
        size = struct.unpack_from(">i", data, offset)[0]
        offset += 4
        if size <= 0 or offset + size > len(data):
            break
        messages.extend(parse_osc_packet(data[offset : offset + size]))
        offset += size
    return messages


def parse_osc_message(data: bytes) -> tuple[str, list[object]] | None:
    try:
        address, offset = read_osc_string(data, 0)
        tags, offset = read_osc_string(data, offset)
        if not tags.startswith(","):
            return None
        args: list[object] = []
        for tag in tags[1:]:
            if tag == "s":
                value, offset = read_osc_string(data, offset)
                args.append(value)
            elif tag == "i":
                args.append(struct.unpack_from(">i", data, offset)[0])
                offset += 4
            elif tag == "f":
                args.append(struct.unpack_from(">f", data, offset)[0])
                offset += 4
            else:
                return None
        return address, args
    except (UnicodeDecodeError, ValueError, struct.error):
        return None


def read_osc_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.index(b"\x00", offset)
    value = data[offset:end].decode("utf-8")
    return value, align4(end + 1)


def align4(value: int) -> int:
    return value + ((4 - value % 4) % 4)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    raise SystemExit(main())
