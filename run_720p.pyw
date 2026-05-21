"""STag tracker — 720p USB camera @ 60fps, double-click to run (.pyw)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from vision_fusion.stag_only import main

    sys.argv = [
        "stag_tracker",
        "--source", "0",
        "--show",
        "--camera-width", "1280",
        "--camera-height", "720",
        "--camera-fourcc", "MJPG",
        "--camera-backend", "msmf",
        "--camera-fps", "60",
        "--camera-exposure", "-4",
        "--predictor", "kalman",
        "--gamma", "0.6",
        "--pass-workers", "12",
    ]

    raise SystemExit(main())
