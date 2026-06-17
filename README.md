# STag Detection

Small Python prototype for detecting STag markers from a camera or video.

The default path is STag-only. YOLO and optical flow are optional tools for more
complex scenes where a coarse ROI or short-term tracking is useful.

## 常用指令速查

先进入项目目录并激活虚拟环境：

```powershell
cd E:\codexbase\c1
.\.venv\Scripts\Activate.ps1
```

如果还没有安装依赖，先执行：

```powershell
pip install -r requirements.txt
```

### 开始捕捉 STag

最常用的实时摄像头捕捉命令：

```powershell
python -m vision_fusion.stag_only --source 0 --show
```

Windows 摄像头建议使用 DirectShow，并指定 640x480 / 60 FPS / MJPG：

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --detect-interval 1 --max-missed 60 --show
```

说明：

```text
--source 0          使用 0 号摄像头；如果有多个摄像头，可试 1、2。
--show              打开实时预览窗口。
--camera-backend    Windows 下可用 dshow 或 msmf。
--detect-interval   每隔多少帧做一次 STag 识别，1 表示每帧识别。
--max-missed        标记短暂丢失后还能保持追踪的帧数。
Q / Esc             关闭预览窗口并退出。
```

### 捕捉视频文件

```powershell
python -m vision_fusion.stag_only --source .\input.mp4 --show
```

### 关闭镜像预览

默认预览是镜像的。如果要使用原始摄像头方向：

```powershell
python -m vision_fusion.stag_only --source 0 --no-mirror --show
```

### 生成 STag 图片

生成单个 ID，例如 `64`：

```powershell
python -m vision_fusion.generate_stag --id 64 --output .\stag_markers\HD17_00064.png --verify
```

生成可打印的多个 marker，并生成 contact sheet：

```powershell
python -m vision_fusion.generate_stag --ids 64 7 12 0 33 --output .\stag_markers --marker-only --label --pixels 448 --sheet --verify
```

打开可视化生成器调参数：

```powershell
python -m vision_fusion.stag_generator_ui
```

### 切角数字 Marker 生成器 (digit_marker_tri)

左上切角定向 + 2×2 大数字 + 加权 mod 11(X 兜底)校验。CLI 批量导出：

```powershell
# 导出 5–25 号到指定目录（闭区间，目录自动建）
python -m vision_fusion.digit_marker_tri --range 5 25 --output out\markers_5_25\
python -m vision_fusion.digit_marker_tri --id 83 --output digit_markers_tri\
```

实时调参（字体 / 字号 / 列距 / 行距 / 粗细 stroke / padding / 切角大小）+ 区间导出 GUI：

```powershell
python -m vision_fusion.digit_marker_tri_ui
```

校验位 = `(1·d₁ + 2·d₂ + 3·d₃) mod 11`，值为 10 时印 `X`（ISBN-10 风格），1000 个 ID 全可用。
左上内角黑三角是方向基准（外框完整方形，仅内部白窗左上被黑三角切角），换位/翻转误读由校验位兜底。

### 屏幕坐标校准

先校准屏幕区域：

```powershell
python -m vision_fusion.calibrate_screen --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --output screen_map.npz
```

按顺序点击屏幕四个角：

```text
top-left, top-right, bottom-right, bottom-left
```

之后带着屏幕映射文件开始捕捉：

```powershell
python -m vision_fusion.stag_only --source 0 --camera-backend dshow --camera-width 640 --camera-height 480 --camera-fps 60 --camera-fourcc MJPG --screen-map screen_map.npz --show
```

注意：校准和运行时要使用相同的镜像设置。默认镜像；如果校准时用了
`--no-mirror`，运行时也要加 `--no-mirror`。

### 发送 TUIO

把 STag 识别结果通过 TUIO 发送到本机 `3333` 端口：

```powershell
python -m vision_fusion.stag_only --source 0 --screen-map screen_map.npz --tuio --tuio-host 127.0.0.1 --tuio-port 3333 --show
```

### 启动手表互动演示

```powershell
python -m vision_fusion.tuio_watch_app --http-host 0.0.0.0 --http-port 8765 --tuio-port 3333
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

没有摄像头/TUIO 发送端时，可以用 demo 模式预览：

```powershell
python -m vision_fusion.tuio_watch_app --demo --http-port 8765
```

### 启动香水互动演示

```powershell
python -m vision_fusion.tuio_perfume_app --http-host 0.0.0.0 --http-port 8776 --tuio-port 3333
```

浏览器打开：

```text
http://127.0.0.1:8776/
```

只用 STag `64` 预览香水 demo：

```powershell
python -m vision_fusion.tuio_perfume_app --demo --demo-ids 64 --http-port 8776
```

### Reactable 香水桌面 (table_app)

从项目根运行：

```powershell
python -m vision_fusion.tuio_table_app          # 接收真实 TUIO (UDP:3333)
python -m vision_fusion.tuio_table_app --demo   # 合成数据预览
```

浏览器打开 http://127.0.0.1:8778/ 。任意带 STag 的香水试香纸片放到桌面，
周围长出信息卡片环；多片间按香调相似度长出连线；把两片凑近自动展开对比。
ID→香水 映射在 `vision_fusion/table_app/id_mapping.js`，换商家改这一个文件。

> 默认仅绑 127.0.0.1（本机）。若需局域网设备访问，加 `--http-host 0.0.0.0`，
> 注意这会开放一个无认证端口。

### YOLO ROI 可选模式

安装 YOLO 依赖：

```powershell
pip install -r requirements-yolo.txt
```

运行 YOLO + STag：

```powershell
python -m vision_fusion.main --source 0 --yolo-model .\best.pt --classes 0 --device 0 --show
```

### 预览颜色含义

```text
green box        当前帧识别到了 STag。
blue box         当前帧没识别到 STag，但用光流记忆继续追踪。
yellow corners   当前 STag 角点观测。
seenmiss         距离上次识别到该 ID 已经过了多少帧。
```

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

Generate STag markers:

```powershell
python -m vision_fusion.generate_stag --id 64 --output .\stag_markers\HD17_00064.png --verify
```

Generate a marker-only PNG like the official preview image:

```powershell
python -m vision_fusion.generate_stag --id 64 --output .\stag_markers\HD17_00064_marker_only.png --marker-only --label --pixels 448 --verify
```

Open the UI generator for live parameter tuning:

```powershell
python -m vision_fusion.stag_generator_ui
```

The UI auto-loads and auto-saves its last parameters in
`stag_generator_settings.json`. Use `Save Params` if you want to force-save the
current knobs immediately.

Generate several printable markers and a contact sheet:

```powershell
python -m vision_fusion.generate_stag --ids 0 64 156 --output .\stag_markers --label --sheet --verify
```

Generate IDs `0..100`:

```powershell
python -m vision_fusion.generate_stag --start 0 --count 101 --output .\stag_markers --marker-only --label --pixels 448 --verify
```

Generate a small synthetic YOLO dataset preview:

```powershell
python -m vision_fusion.generate_synthetic_stag_dataset --count 48 --output .\datasets\stag_synth_preview
```

The default library is HD17, whose valid marker IDs are `0..156`. The generated
PNG includes the white quiet zone around the black marker square; if you use pose
estimation, measure the printed black square for `--marker-size`, not the full
PNG including the white margin.
The generator follows the official STag reference at
`https://github.com/bbenligiray/stag/tree/master/ref/marker%20generator`.

Generator options:

```text
--library 17       STag HD library.
--id 64            One marker ID. Repeat for multiple IDs.
--ids 0 64 156     Marker ID list.
--start 0 --count 10
                   Generate a range.
--all              Generate the whole selected library.
--marker-only      Crop away the white quiet zone for a black-square preview.
--pixels 448       Output image size. In marker-only mode, this is the black square size.
--sheet            Also build a contact sheet PNG.
--verify           Verify generated images with stag.detectMarkers.
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

TUIO watch-table sales client:

```powershell
python -m vision_fusion.tuio_watch_app --http-host 0.0.0.0 --http-port 8765 --tuio-port 3333
```

Open `http://127.0.0.1:8765/` on the touch table. For a preview without a
camera/TUIO sender, add `--demo`.

The sales client treats each `symbol_id` as one watch SKU. The built-in catalog
uses Casio's G-SHOCK military watches collection, including MUDMASTER,
RANGEMAN, MUDMAN, GRAVITYMASTER, 5600/6900 classics, and GD350 vibration-alert
models. It shows a premium watch-store display on the table and records
per-watch sales-interest analytics: table visits, dwell time, touch/focus
interactions, active count, and a combined `interestIndex` ranking. Aggregated
metrics are saved to `watch_analytics.json` by default and are also available at
`http://127.0.0.1:8765/api/analytics`.
Tap a watch/tag in the browser client to open the circular segment menu around
that product. The ring is split into three buttons: photo, details, and color.
A small media window follows the selected tag and lets shoppers browse the
current product image set with carousel arrows and slide dots.

Perfume smart retail demo:

```powershell
python -m vision_fusion.tuio_perfume_app --http-host 0.0.0.0 --http-port 8776 --tuio-port 3333
```

Open `http://127.0.0.1:8776/` on the fragrance table. For a preview without a
camera/TUIO sender, add `--demo`.

If you currently only have STag `64`, use it as the first physical fragrance SKU.
In this demo, `64` maps to `NOCTURNE CEDAR 02`. To preview the same one-tag
setup without the camera/TUIO sender, run:

```powershell
python -m vision_fusion.tuio_perfume_app --demo --demo-ids 64 --http-port 8776
```

To make a multi-bottle physical demo, print one marker per fragrance:

```powershell
python -m vision_fusion.generate_stag --ids 64 7 12 0 33 --output .\stag_markers\perfume --marker-only --label --pixels 448 --sheet
```

The perfume client treats each `symbol_id` as one fragrance SKU. It turns a
physical bottle into an interactive scent profile: fragrance family,
concentration, price, top/heart/base notes, accord strengths, use moments,
pairing ideas, side-by-side comparison, and a session-level scent-memory ranking.
It also keeps a browser-local personal scent session: tried fragrances, dwell
time, repeat visits, LOVE/MAYBE/PASS marks, top accords, style summary, trend
tags, and next-best recommendations.
This demo is designed around the offline advantage of fragrance retail: customers
smell the real product in the store while the screen remembers, explains, and
connects that physical experience to digital content and analytics.

The sender uses TUIO 1.1 `/tuio/2Dobj`:

```text
symbol_id = STag marker ID
session_id = internal track ID
x, y = normalized screen coordinates when --screen-map is set
angle = marker top-edge angle in radians
```

When `--mirror` is enabled, `stag_only` sends mirrored display-space TUIO
coordinates, matching the preview and any screen map captured with the same
mirror setting.

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
