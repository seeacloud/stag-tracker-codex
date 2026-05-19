# AGENTS.md — How to work in this repo

> Conventions for any AI agent (Claude Code, Cursor, Copilot, etc.) or human pairing with one.
> Companion to `SPEC.md`. SPEC says *what* the system does; this says *how* we change it.

## 0. Read first, every session

1. `SPEC.md` — the contract. If your change conflicts with SPEC, update SPEC in the same commit.
2. `README.md` — the user-facing manual. Keep flags / colour key / commands in sync after a change.
3. The module you're touching, fully. Don't grep for symbols and patch in the dark.

If you can't articulate which section of SPEC your change affects, you don't understand the change yet — go read more.

## 1. Project shape

```
vision_fusion/
  models.py           # dataclasses + bbox/track helpers — schema layer, no IO
  stag_detector.py    # wraps stag-python, ROI cropping, optional pose
  optical_flow.py     # LK + forward-backward filter + partial-affine RANSAC
  fusion.py           # FusionTracker — matching, smoothing, dedup
  screen_mapper.py    # 4-point perspective + warped overlays
  tuio_sender.py      # OSC/UDP TUIO 1.1 emitter
  visualization.py    # OpenCV drawing for camera view (colour key lives here)
  stag_only.py        # canonical CLI entry point
  main.py             # optional YOLO+STag+flow CLI
  calibrate_screen.py # interactive 4-corner click tool
  yolo_detector.py    # ultralytics wrapper (only loaded if YOLO path used)
docs/                 # design notes (yolo_stag_training.md etc.)
```

**Layer rules** (don't violate without a SPEC update):

- `models` imports nothing from the other modules. It is the schema. cv2/numpy only.
- `stag_detector` / `optical_flow` / `screen_mapper` / `tuio_sender` depend on `models`, never on each other except where already wired (`tuio_sender` → `screen_mapper`).
- `fusion` orchestrates `optical_flow` and consumes `models`. It does not import cv2 drawing or argparse.
- `visualization` / `screen_mapper` rendering helpers are the only modules allowed to call `cv2.putText` / `cv2.rectangle` / colour constants.
- The CLI entry points (`stag_only.py`, `main.py`, `calibrate_screen.py`) are the only places that touch `argparse`, `cv2.VideoCapture`, `cv2.VideoWriter`, `cv2.imshow`, and `cv2.waitKeyEx`.

If a change wants to violate a layer rule, that's a real architectural decision — surface it, don't sneak it in.

## 2. Style

- **Python 3.11+, `from __future__ import annotations` at the top of every module** that uses postponed evaluation. Match what's already there.
- **Dataclasses with `slots=True`** for any new schema type. Follow the pattern in `models.py`.
- **No new dependencies without explicit asking.** `numpy`, `opencv-python`, `stag-python` are the floor; `ultralytics` is opt-in via `requirements-yolo.txt`. Anything else is a discussion.
- **Type hints on public functions and dataclass fields.** Internal helpers with obvious types can skip them.
- **No comments that restate the code.** A `# Why:` comment for a non-obvious branch (e.g. the `cv2.waitKeyEx` Windows arrow-key constants in `stag_only.handle_key`) is encouraged. Anything that just describes *what* the next line does, delete.
- **Imports**: stdlib, blank line, third-party, blank line, local relative. Sort alphabetically within each group.
- **Errors at the boundary**: `raise SystemExit(...)` for user-facing CLI failures (matches the existing pattern in `stag_only.main`). Never `sys.exit()` from inside library modules.

## 3. Things that are deliberately the way they are

These look like they could be "improved" but were chosen on purpose. Don't refactor them away without asking:

- **STag detection runs on the un-mirrored frame.** Mirroring is display-only. If you "simplify" by mirroring earlier, marker IDs start flipping. (See `stag_only.detection_rois` and `mirror_*_for_display` helpers.)
- **`detect-interval` defaults to 1 in `stag_only` but 5 in `main`.** STag is cheap; YOLO is not. Don't unify.
- **Two parallel CLI entry points (`stag_only` and `main`) instead of subcommands.** The README points users at `stag_only`; `main` exists for completeness. A merge is a SPEC-level change.
- **`_track_rank` prefers `stag` source over `flow`/`predicted`.** When two tracks claim the same marker_id, the freshly-detected one wins. Changing that order is a behavior change.
- **`smooth_alpha` clamp is `[0.05, 1.0]`, not `[0, 1]`.** Zero would freeze the display permanently. Keyboard step is 0.05 for a reason.
- **`frame_index % interval == 0` fires on frame 0.** That's intentional — you want a full search on startup.

## 4. Workflow (Spec-Driven)

1. **Read SPEC** for the relevant section.
2. **Brainstorm/clarify** the change with the user before coding if it's non-trivial. If you're a Claude Code agent with superpowers installed, that's the `brainstorming` skill — use it.
3. **Plan**: list the files you'll touch and the order. If multi-file, write it down.
4. **TDD where it makes sense**: schema math (`models`), TUIO byte layout, screen mapper round-trip — these *should* have tests before the implementation changes. Real-time pipeline behavior (`fusion`, `stag_only`) is hard to unit-test; cover it with a video-file smoke run instead.
5. **Implement.**
6. **Verify** before claiming done — see §6.
7. **Update SPEC.md and README.md** in the same commit if user-visible behavior changed. Stale docs are worse than no docs.

## 5. Don't

- Don't add a new `Track.source` value, CLI flag, or TUIO field without updating SPEC §4 / §3 / §8.
- Don't introduce a global mutable singleton. Pass dependencies into `FusionTracker` / `StagDetector` constructors like the existing code does.
- Don't read `.pen` files with `Read`/`Grep` — use the pencil MCP tools. (Not currently used here, but noted because the global rules require it.)
- Don't commit `*.mp4`, `*.npz`, `*.pt`, `*.onnx`, `__pycache__/`, or `.venv/` — `.gitignore` covers them, but double-check `git status` before staging.
- Don't `git add -A` or `git add .` — stage by name. The repo is small enough.
- Don't mock STag, OpenCV, or the camera in tests for behavior we ship. Unit tests cover schema math; pipeline tests use a small recorded clip checked into a tests fixtures dir (when those exist — see §7).
- Don't bypass safety prompts on destructive git ops (`reset --hard`, `push --force`, `branch -D`). If you think you need them, ask first.

## 6. Verification before claiming done

The minimum bar — the equivalent of "it compiles" for this repo:

```powershell
.\.venv\Scripts\python.exe -m vision_fusion.stag_only --help
.\.venv\Scripts\python.exe -m vision_fusion.calibrate_screen --help
.\.venv\Scripts\python.exe -m vision_fusion.main --help
```

All three must exit 0. Then a no-camera smoke run with a recorded video:

```powershell
.\.venv\Scripts\python.exe -m vision_fusion.stag_only --source .\fixtures\smoke.mp4 --max-frames 120 --log-every 30
```

(The fixture doesn't exist yet — first time you need it, record one and check it in *small*, e.g. ≤2 MB, < 5 s, 480p. Update `.gitignore` to allow `fixtures/*.mp4` if needed.)

When the change touches camera/preview behavior, run the canonical 60 FPS-oriented command from `README.md` for ~10 seconds and eyeball that:

- The status bar updates smoothly.
- A marker shows green box → blue box → green box across an occlusion.
- `active` and `observed` numbers in the FPS log look sane.

Save a screenshot to `docs/test-screenshots/<UTC-timestamp>-<change>.png` and reference it in the commit / PR. The global rule "测试必须自动化 + 留档" applies here.

If you can't run a camera, say so explicitly. Don't claim "tested" because `--help` exited 0.

## 7. Tests (status: missing, planned)

There are no automated tests in the repo today. This is the biggest debt. When you next touch a module, add the matching test:

- `tests/test_models.py` — `clip_bbox`, `bbox_iou`, `bbox_from_points`, `append_track_history` trimming at `TRACK_HISTORY_LIMIT`.
- `tests/test_screen_mapper.py` — round-trip a known quad through `from_points` → `save` → `load` → `transform_points`, assert ≤1px reprojection error.
- `tests/test_tuio_sender.py` — golden-byte test for `osc_message` and `osc_bundle` on a known input.
- `tests/test_fusion.py` — synthetic frames + fake observations to assert `marker_id` continuity through a 5-frame "stag → flow → stag" cycle.

Run with `python -m pytest`. Add `pytest` to `requirements.txt` *only* when the first test lands.

## 8. Commits & PRs

- Conventional, terse commit subjects: `stag_only: …`, `fusion: …`, `docs: …`. Match `git log`'s existing tone (`Color tag trajectories by history state`, `Initial STag tracking prototype`).
- One concern per commit. Don't bundle a refactor with a behavior change.
- PR description: *what* changed, *why*, and *how it was verified* (with the screenshot path from §6 if applicable).
- Push to a feature branch, never directly to `main`. Confirm with the user before opening a PR — don't auto-open.

## 9. When in doubt

Ask. The cost of a clarifying question is one message; the cost of a wrong refactor that gets shipped is hours of untangling. SPEC and this file are not exhaustive — they're the floor.
