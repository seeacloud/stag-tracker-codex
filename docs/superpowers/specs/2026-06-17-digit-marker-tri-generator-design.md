# 数字 Marker 生成器 `digit_marker_tri` — 设计文档

- 日期：2026-06-17
- 阶段：marker 生成工具
- 状态：设计已与用户敲定，待写实现计划

## 1. 背景与目标

现有数字 marker 生成器有三版（`digit_marker` v1 黑框+圆点+单行；v2 圆点+2×2；
`digit_marker3` 缺口边框+2×2），方向标记与校验位都有历史包袱：v1/v2 用左上
圆点，v3 用边框缺口；校验位一律是"3 位 ID 各位求和 mod 10"。求和校验对数字
顺序不敏感（083/038/308 校验位相同），抓不住换位错；圆点是内部特征，IR 背投
模糊下易丢失。

本设计新建一版 `digit_marker_tri`，目标：

- 左上**切角(chamfer)**定向：改变外轮廓（方形→五边形），是轮廓级特征，比
  内部圆点更抗模糊。
- **加权 mod 11 + X** 校验：质数模数，单点错与换位错 100% 可检。
- 暴露一组排版旋钮（字体/字号/列距/行距/粗细/padding/切角大小），CLI 与实时
  预览 GUI 都能调。

不碰 v1/v2/v3 及其现有 UI（守住 working state）。

## 2. 关联决策（见 memory `digit-marker-rotation-orientation`）

- 方向**必须**靠物理标记，不能"暴力试 4 朝向 + 校验位猜"——0695 倒 180° 读成
  5690 也能过校验，会误判。切角是本版选定的物理方向基准。
- 已否决 hex 扩 ID（A–F 引爆字形混淆 8↔B/0↔D/6↔G，且破坏 mod 11 的质数性）；
  以后扩 ID 走"加十进制格子"，不加每格符号数。

## 3. 架构（3 个新文件）

```
vision_fusion/
  digit_marker_tri.py        # 核心函数 + CLI
  digit_marker_tri_ui.py     # Tkinter 实时预览 GUI（import 核心）
vision_fusion/tests/
  test_digit_marker_tri.py   # pytest 单测
```

分层：纯核心函数（无 IO、可单测）← 上层 CLI（批量/单张）与 GUI（实时预览 +
存档）。GUI 参数存 `digit_marker_tri_settings.json`（不与旧 UI 的
`digit_marker_settings.json` 冲突）。

### 核心 API

```python
def checksum_char(marker_id: int) -> str: ...
def generate_marker_tri(
    marker_id: int, *, pixels=600, border_ratio=0.07, chamfer_ratio=0.18,
    pad_ratio=0.06, col_gap_ratio=0.34, row_gap_ratio=0.34,
    font_path="C:/Windows/Fonts/consolab.ttf", font_size_ratio=0.28,
    stroke_ratio=0.0,
) -> np.ndarray: ...
```

返回灰度 `ndarray`（uint8，背景 255 黑墨 0，与 v3 一致，方便 `Image.fromarray` 存 PNG）。

## 4. 数据层：校验位

```python
def checksum_char(marker_id: int) -> str:
    d = f"{marker_id:03d}"
    c = (1 * int(d[0]) + 2 * int(d[1]) + 3 * int(d[2])) % 11
    return "X" if c == 10 else str(c)
# 083 → 1·0 + 2·8 + 3·3 = 25, 25 % 11 = 3 → "3" → marker 文本 "0833"
```

- 定义域 ID 0..999，全部可用（校验=10 印 `X`，ISBN-10 风格）。
- marker 文本 = `f"{id:03d}"` + `checksum_char(id)`，共 4 字符。
- 阅读序（切角锁定 TL 起）：**TL=d₁, TR=d₂, BL=d₃, BR=校验**。

## 5. 几何与参数

```
┌╱────────────┐   outer 黑框，左上角切掉三角(chamfer)
│   d₁     d₂  │
│              │   ← 行距 row_gap（两行心间距）
│   d₃     检  │     列距 col_gap（两列心间距）
└──────────────┘   黑框 border_ratio；数字块四周 pad_ratio
```

绘制步骤（PIL，灰度 `L`，背景 255）：

1. 画满黑 `[0,0,p-1,p-1]` → 挖白内部 `[b,b,p-1-b,p-1-b]`（`b=border_ratio·p`），
   形成黑边框 ring。
2. 切角：左上画白三角 polygon `[(0,0),(ch,0),(0,ch)]`（`ch=chamfer_ratio·p`），
   切掉外角 → 黑区轮廓成五边形（唯一开口在左上）。
3. content box = 内框四周再缩 `pad_ratio·p`。
4. 2×2 格心：以 content box 中心为基准，列心相距 `col_gap_ratio·p`、行心相距
   `row_gap_ratio·p`（四心在中心 ±gap/2）。
5. 字号 = `font_size_ratio·p`；`font = truetype(font_path, size)`，失败回退
   `load_default`。
6. 每格 `draw.text` 居中绘制单字符，`stroke_width = round(stroke_ratio·font_size)`、
   `stroke_fill=0` 仿粗。

| 参数 | 含义 | GUI 范围 | 默认 |
|---|---|---|---|
| `pixels` | 输出边长 px | 200–1200 (spinbox) | 600 |
| `border_ratio` | 黑框厚度 | 0.03–0.15 | 0.07 |
| `chamfer_ratio` | 切角沿两边长度 | 0.05–0.35 | 0.18 |
| `pad_ratio` | 数字块四周留白 | 0.0–0.20 | 0.06 |
| `col_gap_ratio` | 两列心间距 | 0.15–0.55 | 0.34 |
| `row_gap_ratio` | 两行心间距 | 0.15–0.55 | 0.34 |
| `font_size_ratio` | 字号 | 0.10–0.45 | 0.28 |
| `stroke_ratio` | 仿粗（占字号比） | 0.0–0.15 | 0.0 |
| `font_path` | 字体文件 | 系统字体下拉 | consolab.ttf |

（默认值实现时按截图微调，保证四字不重叠、不出框。）

## 6. GUI（`digit_marker_tri_ui.py`）

沿用 `digit_marker_ui` 模式：左侧控件（ID spinbox、字体 combobox =
`list_system_fonts()`、各 ratio 滑块带数值回显、stroke 滑块、pixels spinbox）；
右侧 400×400 实时预览（`cv2.resize` + `ImageTk`）。

导出区（**ID 可选 + 批量到指定目录**）：

- **ID spinbox**（0–999）任选单个，实时预览 + "保存当前" 按钮导出这一个。
- **批量导出**：`起始 ID` / `结束 ID` 两个输入框 + `输出目录` 字段 + "批量导出"
  按钮，把 `[start, end]` 闭区间全部用当前排版参数生成并写入该目录（目录不存在
  自动建）。范围不写死 0–99。

状态栏显示 `ID / 校验 / 文本` 与导出结果。参数（含 start/end/输出目录）存
`digit_marker_tri_settings.json`。

## 7. CLI（`digit_marker_tri.py`）

```bash
# 单个
python -m vision_fusion.digit_marker_tri --id 83 --output digit_markers_tri/
# 批量导出 5–25 号到指定目录（闭区间，目录自动创建）
python -m vision_fusion.digit_marker_tri --range 5 25 --output out/markers_5_25/
# 全量
python -m vision_fusion.digit_marker_tri --range 0 999 --output digit_markers_tri/
```

flag：`--id`（单个） / `--range START END`（闭区间批量） / `--output`（目标目录，
不存在则建） / `--pixels` + 上述各 ratio（默认同表）。`--id` 与 `--range` 二选一；
都不给默认 `--range 0 99`。输出文件名 `digit_{id:03d}_{check}.png`（带校验位便于
人眼核对）。

## 8. 测试与验证

单测（pytest，`test_digit_marker_tri.py`）：

- **校验位**：`083→"3"` 等若干已知值；构造一个 `c==10` 的 ID 验证返回 `"X"`；
  遍历 0..999 全部返回单字符且 ∈ `{0-9,X}`。
- **布局不变量**：`generate_marker_tri` 返回 `shape=(pixels,pixels)` uint8；
  左上切角三角形心点像素为白(255)（方向标记真切了）；四个格心邻域存在暗像素
  (<128)（数字真画了）；增大 `stroke_ratio` 时暗像素总数单调不减（仿粗生效）。

可视存档：CLI 生成 `0/83/999` 等几张 + GUI 截图，存
`docs/test-screenshots/<时间戳>/`，README 给路径供人工核查切角/间距/粗细。

## 9. 范围边界（YAGNI）

不做：拼版 sheet、PDF、非方形、彩色、二维码兜底、**检测/解码端改动**（解码端
适配新切角+新校验另列任务，本 spec 只管生成）。不碰 v1/v2/v3 及其现有 UI。

