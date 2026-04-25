# STag Detection

Small Python prototype for detecting STag markers from a camera or video.

The default path is STag-only. YOLO and optical flow are optional tools for more
complex scenes where a coarse ROI or short-term tracking is useful.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`stag-python` may need a platform-specific wheel. If installation fails, install the STag
Python binding recommended for your environment and make sure `import stag` works.

## Run

STag camera with memory tracking:

```powershell
python -m vision_fusion.stag_only --source 0 --show
```

Mirrored camera view:

```powershell
python -m vision_fusion.stag_only --source 0 --show
```

Disable mirrored preview:

```powershell
python -m vision_fusion.stag_only --source 0 --no-mirror --show
```

STag-only 60 FPS-oriented camera run:

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --detect-interval 1 --max-missed 60 --show
```

Video:

```powershell
python -m vision_fusion.stag_only --source .\input.mp4 --show
```

Screen calibration:

```powershell
python -m vision_fusion.calibrate_screen --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --output screen_map.npz
```

Click the display corners in this order:

```text
top-left, top-right, bottom-right, bottom-left
```

Then run with the saved screen map:

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --screen-map screen_map.npz --show
```

The warped screen view uses the clicked screen shape. It does not force 16:9. If
you want a fixed output size, pass both `--width` and `--height` during calibration.
Mirroring is enabled by default. Use the same mirror setting for calibration and
runtime; pass `--no-mirror` to both commands if you want raw camera orientation.

Optional YOLO ROI mode:

```powershell
pip install -r requirements-yolo.txt
python -m vision_fusion.main --source 0 --yolo-model .\best.pt --classes 0 --device 0 --show
```

Useful options:

```text
--roi-padding 12          Expand each STag detection ROI.
--stag-library 17         STag HD library number.
--mirror / --no-mirror    Flip preview/output horizontally; enabled by default.
--screen-map screen.npz   Load display perspective calibration.
--screen-output out.mp4   Save the warped screen view.
--no-screen-roi           Search the full camera frame even when screen-map exists.
--tuio                    Send TUIO 1.1 /tuio/2Dobj over UDP.
--tuio-host 127.0.0.1     TUIO target host.
--tuio-port 3333          TUIO target port.
--detect-interval 1       Run STag detection every N frames.
--reacquire-interval 30   Full-frame search cadence while tracking.
--fallback-full-interval 5
                           Add a full-screen/screen-map ROI while tracking.
--search-padding 80       Search area around remembered tracks.
--max-missed 60           Keep tracks alive through temporary optical-flow misses.
--visual-hold 6           Prevent brief stag/flow visual flicker.
--smooth-alpha 0.35       Display smoothing weight; lower is steadier but laggier.
--smooth-deadband 1.5     Ignore tiny display movements in pixels.
--smooth-snap 70          Snap to raw track after a large movement.
--smooth-step 0.05        Keyboard step for live smoothing changes.
--no-memory               Disable optical-flow memory tracking.
--marker-size 0.08        Marker side length in meters, enables pose if calibration is set.
--calibration calib.npz   NPZ with camera_matrix and dist_coeffs arrays.
--output tracked.mp4      Save annotated video.
--max-frames 300         Stop after N frames for benchmarking.
--camera-backend dshow   Use DirectShow camera backend on Windows.
```

## Performance Notes

For pure STag detection, use the dedicated STag-only command. On the test machine,
the camera reported 60 FPS but actually delivered about 32 FPS through OpenCV.
The software path was not the primary bottleneck.

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --detect-interval 1 --max-missed 60 --max-frames 300
```

If the marker flickers under blur, keep memory tracking on and increase the search
window:

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --search-padding 120 --max-missed 90 --show
```

Display colors:

```text
green box        STag was recognized on this frame.
blue box         Memory/optical-flow tracking while STag is not currently recognized.
yellow corners   Current STag corner observation only.
seenmiss         Frames since the marker ID was last recognized.
```

When the marker is covered, `seenmiss` rises even if optical flow can still follow
the covering object. The track is removed when `seenmiss` or optical-flow `miss`
exceeds `--max-missed`.

If the marker box jitters, lower the smoothing alpha and raise the deadband:

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --smooth-alpha 0.2 --smooth-deadband 2.5 --show
```

If it feels too sluggish while moving fast, increase `--smooth-alpha` toward `0.5`
or reduce `--smooth-snap`.

Live keyboard tuning while the preview window is focused:

```text
Up / W       more stable, lower smooth-alpha
Down / S     more responsive, higher smooth-alpha
Left / A     lower deadband
Right / D    higher deadband
Q / Esc      quit
```

TUIO output:

```powershell
python -m vision_fusion.stag_only --source 0 --screen-map screen_map.npz --tuio --tuio-host 127.0.0.1 --tuio-port 3333 --show
```

The sender uses TUIO 1.1 `/tuio/2Dobj`:

```text
symbol_id = STag marker ID
session_id = internal track ID
x, y = normalized screen coordinates when --screen-map is set
angle = marker top-edge angle in radians
```

YOLO training notes for a one-class `stag` detector are in
`docs/yolo_stag_training.md`.

If you still want YOLO + STag + optical flow, use:

```powershell
python -m vision_fusion.main --source 0 --yolo-model .\best.pt --classes 0 --device 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --detect-interval 10 --yolo-imgsz 416 --yolo-half --show
```

## Calibration File

Pose estimation is optional. Save calibration like this:

```python
import numpy as np

np.savez(
    "calib.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
)
```

## Runtime Behavior

`vision_fusion.stag_only` runs STag and keeps a memory track for each marker. When
STag temporarily fails because of blur or motion, Lucas-Kanade optical flow predicts
the remembered marker bbox and corners, then STag searches near that remembered
position to reacquire the ID.
If `--screen-map` is set, full searches are restricted to the display region by
default and a warped `Screen view` window shows the display as a flat rectangle.
`vision_fusion.main` keeps the previous YOLO + optical-flow fusion path available
for ROI detection and short-term tracking.
