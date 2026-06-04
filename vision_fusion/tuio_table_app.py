from __future__ import annotations

import argparse
import math
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from .tuio_watch_app import (
    SharedTuioState,
    clamp01,
    make_handler,
    run_tuio_udp_listener,
)


ASSET_DIR = Path(__file__).with_name("table_app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reactable perfume table demo (phase A).")
    parser.add_argument("--tuio-host", default="0.0.0.0", help="UDP address to listen on for TUIO.")
    parser.add_argument("--tuio-port", type=int, default=3333, help="UDP port to listen on for TUIO.")
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP host for the browser app.")
    parser.add_argument("--http-port", type=int, default=8778, help="HTTP port for the browser app.")
    parser.add_argument("--event-hz", type=float, default=30.0, help="SSE update frequency.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic objects for preview.")
    parser.add_argument(
        "--demo-ids",
        nargs="+",
        type=int,
        default=[64, 7, 12],
        help="Synthetic symbol IDs to animate when --demo is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop = threading.Event()
    state = SharedTuioState(analytics_path=None)

    threads = [
        threading.Thread(
            target=run_tuio_udp_listener,
            args=(args.tuio_host, args.tuio_port, state, stop),
            daemon=True,
        )
    ]
    if args.demo:
        threads.append(threading.Thread(target=run_demo, args=(state, stop, args.demo_ids), daemon=True))
    for thread in threads:
        thread.start()

    handler = make_handler(state, max(args.event_hz, 1.0), asset_dir=ASSET_DIR)
    server = ThreadingHTTPServer((args.http_host, args.http_port), handler)
    print(f"TUIO table app: http://{args.http_host}:{args.http_port}/")
    print(f"Listening for /tuio/2Dobj on udp://{args.tuio_host}:{args.tuio_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


def run_demo(state: SharedTuioState, stop: threading.Event, demo_ids: list[int]) -> None:
    frame = 0
    start = time.perf_counter()
    demo_ids = demo_ids or [64]
    while not stop.is_set():
        t = time.perf_counter() - start
        objects: list[dict[str, float | int]] = []
        for index, symbol_id in enumerate(demo_ids):
            # 周期性把前两片拉近再分开，便于验证凑近对比手势。
            converge = (math.sin(t * 0.25) + 1) / 2  # 0..1
            base_x = 0.5 + (index - (len(demo_ids) - 1) / 2) * 0.22
            x = base_x + (0.5 - base_x) * converge * 0.7
            y = 0.5 + math.sin(t * 0.2 + index) * 0.06
            objects.append(
                {
                    "sessionId": index + 1,
                    "symbolId": symbol_id,
                    "x": clamp01(x),
                    "y": clamp01(y),
                    "angle": t * 0.3 + index,
                    "xVelocity": 0.0,
                    "yVelocity": 0.0,
                    "angleVelocity": 0.3,
                    "motionAccel": 0.0,
                    "rotationAccel": 0.0,
                    "lastSeen": time.time(),
                }
            )
        state.set_demo_objects(frame, objects)
        frame += 1
        time.sleep(1 / 30)


if __name__ == "__main__":
    raise SystemExit(main())
