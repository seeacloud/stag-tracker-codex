# STag Tracker — Specification

> Source of truth for what this project does. README is the user manual; this is the contract.
> If code and SPEC disagree, fix one of them — don't let them drift.

## 1. Purpose

Detect [STag](https://github.com/bbenligiray/stag) fiducial markers in camera or video input, keep a stable per-marker track through brief occlusion/blur using optical flow, and optionally:
- Warp the camera view to a flat "screen" plane via 4-point perspective calibration.
- Emit marker pose to external software via TUIO 1.1 over UDP.
- Use a YOLO detector as a coarse ROI prefilter for crowded scenes.

Default path is **STag-only with optical-flow memory** (`vision_fusion.stag_only`). YOLO fusion (`vision_fusion.main`) is a parallel path kept available, not the recommended one.

## 2. Scope

In scope:
- Single-machine, single-camera real-time prototype.
- Windows-first (DirectShow / MSMF backends, MJPG fourcc) but not Windows-only.
- Python 3.11+, CPU-only by default. GPU only required if the optional YOLO path is used.

Out of scope (don't add without an explicit ask):
- Multi-camera fusion / stereo.
- Network streaming of video frames.
- A GUI beyond the OpenCV preview windows.
- A persistent database / web service / mobile client.
- Replacing OpenCV with a different vision stack.

## 3. Entry points

| Module                              | Purpose                                                       | Stable? |
| ----------------------------------- | ------------------------------------------------------------- | ------- |
| `vision_fusion.stag_only`           | Recommended runtime: STag + optical-flow memory + TUIO/screen | Yes     |
| `vision_fusion.calibrate_screen`    | Click 4 screen corners, save `screen_map.npz`                 | Yes     |
| `vision_fusion.main`                | Optional YOLO + STag + flow fusion                            | Kept    |

CLI flags are the public interface. Any flag rename is a breaking change — keep the old flag working or bump a version.

## 4. Data contract

These types live in `vision_fusion/models.py` and are the boundary between subsystems. Treat them as the schema.

- **`BBox = (x, y, w, h)`** — int pixels, top-left origin, never negative width/height after `clip_bbox`.
- **`StagObservation`** — one detection from STag in this frame: `marker_id`, 4×2 `corners` (float32, image space), `bbox`, optional `pose`.
- **`Track`** — persistent identity across frames. Carries the *raw* `bbox`/`corners` plus *display-smoothed* `display_bbox`/`display_corners`, plus history (capped at `TRACK_HISTORY_LIMIT = 64`) for trajectory rendering and TUIO velocity.
- **`Pose`** — `(rvec, tvec)` from `cv2.solvePnP(IPPE_SQUARE)`. Only populated when both `--marker-size` and `--calibration` are provided.
- **`CameraCalibration`** — `camera_matrix`, `dist_coeffs` loaded from NPZ.

### Track source labels (invariant)

`track.source` is one of `{"init", "stag", "flow", "predicted"}`. Visualization and TUIO logic depend on this label — do not introduce new values without updating `visualization.track_state_color` and the README colour key.

| Source       | Meaning                                                  |
| ------------ | -------------------------------------------------------- |
| `init`       | Just created from a measurement, no history yet          |
| `stag`       | Updated this frame from an actual STag detection         |
| `flow`       | Updated by optical flow (LK + partial-affine RANSAC)     |
| `predicted`  | LK failed; bbox dead-reckoned by last velocity           |

`detection_missed` counts frames since the last `stag` update. `missed` counts frames since the last successful flow update.

## 5. Pipeline (stag_only, the canonical path)

Per frame, in order:

1. **Capture** — `cv2.VideoCapture` with `BUFFERSIZE=1`. On Windows, prefer `--camera-backend dshow --camera-fourcc MJPG` for high-FPS USB cameras.
2. **Greyscale** — single `cv2.cvtColor` shared by detection and flow.
3. **Predict** (if memory enabled and previous gray exists): run optical flow on each existing track, increment `detection_missed`, smooth `display_bbox`/`display_corners`, merge duplicate-marker tracks, drop stale ones (> `--max-missed`).
4. **Detect** (every `--detect-interval` frames): build ROIs (see §6), run STag, dedupe by `(marker_id, top-left)` then by IoU.
5. **Fuse** (`FusionTracker.update`): match each observation to an existing track first by `marker_id`, then by IoU (threshold `0.25`); update or create. Tracks with the same `marker_id` are collapsed via `_track_rank` keeping the freshest STag-sourced one.
6. **Render** — annotated camera view (mirrored by default), optional warped "Screen view", optional MP4 writer.
7. **TUIO emit** (if enabled) — see §8.
8. **Log** — every `--log-every` frames, print `frame=… fps_wall=… fps_loop=… active=… observed=…`.

## 6. ROI strategy

`stag_only.detection_rois` decides where STag runs each detect frame. This is the main lever that keeps it real-time:

- **No tracks yet** *or* `frame_index % --reacquire-interval == 0` → full-frame search (or screen-map source bbox if `--screen-map` set and `--no-screen-roi` not set).
- **With tracks** → one ROI per track, expanded by `--search-padding`.
- **Periodic safety net** — every `--fallback-full-interval` frames, *also* add a full-frame/screen-map ROI on top of the per-track ROIs, so a brand-new marker entering the scene gets picked up without waiting for the next full reacquire.
- **Mirroring** — internal detection runs on the un-mirrored frame. Only the displayed bbox/corners are mirrored (`mirror_*_for_display`). Never mirror before STag detection.

## 7. Smoothing

`FusionTracker._smooth_bbox` / `_smooth_points`:

- `smooth_alpha == 1.0` → no smoothing (raw track).
- `0 < smooth_alpha < 1` → exponential moving average toward target.
- Movement under `smooth_deadband` pixels (center delta and size delta) is ignored.
- Movement over `smooth_snap` pixels snaps directly to target (avoids lag on a fast flick).
- Live keys: ↑/W decrease alpha (steadier), ↓/S increase (livelier), ←/A decrease deadband, →/D increase. Range clamps in `stag_only.handle_key`.

Display-time outputs (`display_bbox`, `display_corners`) are what the user sees and what TUIO sends. Raw `bbox`/`corners` are what the matcher and ROI builder consume. Don't swap them.

## 8. TUIO 1.1 contract

Implemented in `vision_fusion/tuio_sender.py`. One UDP packet per frame, OSC bundle containing:

```
/tuio/2Dobj source <source-name>
/tuio/2Dobj set <session_id> <symbol_id> <x> <y> <angle> <vx> <vy> 0.0 0.0 0.0   # repeated per object
/tuio/2Dobj alive <session_id...>
/tuio/2Dobj fseq <frame>
```

- `session_id` = internal `track.track_id` (stable across frames for the same identity).
- `symbol_id` = STag `marker_id`.
- `x, y` ∈ `[0, 1]`, normalized to **screen-mapper output size** if `--screen-map` is set, otherwise to the camera frame.
- `angle` = `atan2(corners[1] - corners[0])` in radians (top-edge angle).
- Velocities are 0 when a screen mapper is active (perspective warp invalidates linear pixel velocity); otherwise `vx, vy` are the LK median deltas normalized by frame size.
- Tracks without `marker_id` (orphan flow tracks from YOLO path) are skipped.
- On shutdown, send an empty alive list so receivers drop sessions cleanly.

## 9. Files & artifacts

- **`screen_map.npz`** — written by `calibrate_screen`. Keys: `source_points` (4×2 float32, image space), `output_size` (int32 [w,h]), `camera_to_screen`, `screen_to_camera` (3×3 perspective matrices). Click order is `top-left, top-right, bottom-right, bottom-left` — order is load-bearing, do not reorder.
- **`calib.npz`** — user-provided. Keys: `camera_matrix`, `dist_coeffs`. Used only when `--calibration` and `--marker-size` are both passed.
- **`*.mp4` / `*.avi`** — `--output` writes annotated video; `--screen-output` writes the warped view. Codec is `mp4v`. FPS falls back to 30 if the source reports < 1.

`*.npz`, `*.mp4`, `*.pt` are gitignored; never commit them.

## 10. Performance targets

- **stag_only on a typical USB cam at 640×480 MJPG**: 25–32 fps_wall, fps_loop up to ~33. Software is not the bottleneck — the camera driver is.
- **stag_only with screen-map ROI**: should be no slower than full-frame, often faster.
- **YOLO+STag (`main`)** at 1280×720, `--detect-interval 10`, `--yolo-imgsz 416`, `--yolo-half`: should not drop below ~20 fps on a modest GPU.

If a change drops fps_loop on the canonical command (§ README "60 FPS-oriented") by more than ~10% on the same hardware, treat it as a regression.

## 11. Acceptance checks

For any non-trivial change, the following must still hold:

1. `python -m vision_fusion.stag_only --help` exits 0 and lists every flag in §3-§7 above.
2. `python -m vision_fusion.stag_only --source 0 --max-frames 60 --log-every 30` runs to completion, prints two FPS lines, exits 0.
3. With a marker in view, `active` ≥ 1 within the first 60 frames and the on-screen state cycles `stag` → `hold`/`flow` → `stag` when the marker is briefly covered.
4. Adding `--screen-map screen_map.npz` opens a "Screen view" window of the calibrated output size; closing the source window closes both.
5. Adding `--tuio` emits ≥ 1 UDP packet per frame to the configured host:port (verify with `nc -ul 3333` / `Test-NetConnection` style probe).
6. `--no-mirror` produces the raw camera orientation in *both* preview and `--output` video; mirror state must match what was used at calibration.

When you can't run the camera tests in CI, at minimum run #1, #2 with a recorded video file as `--source`, and the unit-level checks below.

## 12. Invariants (don't break these silently)

- STag detection always sees the **un-mirrored** frame.
- `Track.source` ∈ the closed set in §4.
- `display_*` fields are derived; nothing else writes to them outside `_stabilize_track`.
- `marker_id` is the identity for a track once assigned. Two live tracks with the same `marker_id` is a bug — `_merge_duplicate_marker_tracks` enforces this.
- Frame indexing starts at 0 and increments after rendering, so `frame_index % interval == 0` fires on the first frame.
- TUIO `x, y` are in `[0, 1]` (clamped). Anything outside is a bug, not a feature flag.

## 13. Open questions / known gaps

- No automated tests exist yet. A `tests/` directory with at least: a `models` math suite (bbox / IoU / history trim), a `screen_mapper` round-trip (`from_points → save → load → transform_points`), and a TUIO byte-layout golden test, is the minimum to call this production-ready.
- `vision_fusion.main` (YOLO path) shares a lot of plumbing with `stag_only` but copy-pastes the camera-config / writer code. If both paths stay long-term, factor into a shared `runtime/` module.
- No Linux/macOS smoke run has been recorded — the `dshow` / `MSMF` flags are Windows-only, but the rest should work; verify before claiming cross-platform.
