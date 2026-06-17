# 数字 Marker 解码器 `digit_detect_tri` — 设计文档

- 日期：2026-06-17
- 阶段：digit_marker_tri 配套解码端
- 状态：设计已与用户敲定，待写实现计划

## 1. 背景与目标

`digit_marker_tri`（内角黑三角定向 + 加权 mod11+X 校验）的 marker 已能生成。现有
两个解码器都对不上它：

- `digit_detect.py`：YOLO-OBB 定位 → `orient_by_dot`（圆点模板）→ RapidOCR 读 →
  校验 `sum%10`。
- `digit_detect3.py`：OBB → `orient_by_notch`（缺口）→ 同 `sum%10`。

实拍验证（用户照片，`digi_marker_obb.pt` 跑出）：YOLO-OBB **定位很准**、RapidOCR
**4 位含 X 都读对了**（007X/0204/0197… 经加权 mod11 逐个核对一致），但全部显示
"?"。根因三处：① `recognize` 用 `isdigit` 把 `X` 滤掉，"007X"→"007" 只剩 3 位；
② 校验用旧 `sum%10`，与加权 mod11 对不上；③ 定向 `orient_by_dot` 找圆点，tri 是
黑三角，不匹配。

目标：新建 `digit_detect_tri.py`，复用"定位 + 读数 + 追踪"，换掉"定向 + 校验 +
保留 X"，把实拍里的 "?" 变成确认 ID。不碰 `digit_detect.py`/`digit_detect3.py`。

## 2. 关联决策

- 见 memory `digit-marker-rotation-orientation`（定向靠物理标记，不试 4 朝向撞校验）
  与 `digit-detect-rapidocr-pipeline`（RapidOCR rec-only 读 4 位、MarkerTracker 加权
  投票，都可复用）。
- **定向不用模板匹配**：实拍模糊会糊掉三角的形状/边缘，`matchTemplate` 靠形状会
  失配。黑三角的价值是"往一个角砸进一大块黑色质量"——模糊成灰团后，那个角的
  **积分暗度**仍明显高于其余三角。所以定向测**暗度**，不测形状。这也是 tri 用三角
  取代圆点的根本原因（圆点暗度信号太弱，memory 记只有 47%）。

## 3. 架构（新文件 `vision_fusion/digit_detect_tri.py`）

**复用**（import，不复制）：
- from `.digit_detect`：`order_corners`、`warp_square`、`draw_corners`、
  `marker_up_vector`、`id_anchor`、`MarkerTracker`。
- from `.digit_marker_tri`：`checksum_char`。
- 外部：`ultralytics.YOLO`、`rapidocr_onnxruntime.RapidOCR`、cv2、numpy。

**新写**三块（旧解码器对不上的部分）：`orient_by_triangle` / `find_triangle_corner`、
`DigitRecognizerTri`、`main()`。

数据流：摄像头帧 → CLAHE → YOLO-OBB(`digi_marker_obb.pt`) 找 marker 四角 →
`warp_square` 透视矫正成 SxS → `orient_by_triangle` 旋正 → 裁上下两行 → RapidOCR
rec-only → 保留 X 取 4 字符 → 加权 mod11+X 校验 → 叠加 ID/方向（可选 MarkerTracker
投票）。

> **YOLO 模型注意**：`digi_marker_obb.pt` 是用**旧 marker**（圆点/缺口设计）训练的，
> 与当前 tri（内角黑三角）**不是同一种**。YOLO-OBB 主抓"黑方框 + 2×2 数字"这个大
> 特征，角上小三角对它影响小，**可能**直接泛化（实拍照片里框得尚可），但这是未验证
> 假设。本 plan 把"在 tri marker 上验证现有模型的检出率"列为显式步骤；若检出不足，
> **另起子项目**合成 tri 训练数据重训 OBB（复用 `generate_marker_tri` + `gen_notch_yolo.py`
> 的数据合成/标注套路 → `models/tri_marker_obb.pt`），`--model` 可切换，解码逻辑不变。

## 4. 三角定向（内角暗度）

`find_triangle_corner(square) -> (corner:int, conf:float)`：

warp 后 marker 充满 SxS，外圈黑边框厚 `b≈border_ratio*S`，黑三角在某个内角，约
占 `[b : b+cut, b : b+cut]`（`cut≈chamfer_ratio*S`）的对角半区。算法：

1. 在 4 个**内角缝隙**各取一块贴角顶的小patch（在边框内侧、数字够不到处），
   patch 边长 `k = int(S * 0.12)`，起点紧贴内边框：
   - TL: `square[b:b+k, b:b+k]`
   - TR: `square[b:b+k, S-b-k:S-b]`
   - BR: `square[S-b-k:S-b, S-b-k:S-b]`
   - BL: `square[S-b-k:S-b, b:b+k]`
2. 每块暗度 = `(255 - patch).mean()`（积分量，天然低通、抗模糊）。
3. `corner = argmax(暗度)`；`conf = (最暗 - 次暗) / (最暗 + 1e-6)`（相对 margin）。

`orient_by_triangle(square, min_conf=0.15)`：取 `find_triangle_corner`，conf 不足
返回 `(square, False)`（定向存疑，上层画 "?"）；否则按角旋到左上并返回 `(rot, True)`。
旋转映射同 `orient_by_dot`：0=不转，1(TR)=ROTATE_90_COUNTERCLOCKWISE，
2(BR)=ROTATE_180，3(BL)=ROTATE_90_CLOCKWISE。

> 抗模糊三要点：积分暗度低通；patch 贴角顶取小，只罩三角、避开各格数字笔画
> （免得重墨数字盖过带三角的角）；margin 门槛不够就标 "?"，宁可不定向也不乱定向。

## 5. 识别与校验（`DigitRecognizerTri`）

`recognize(square, min_conf=0.5) -> (marker_id:int, conf:float)`：

1. `oriented, ok = orient_by_triangle(square)`；`ok=False` 直接返回 `(-1, 0)`。
2. 裁上下两行（**全宽两列**，因 tri 四格都是数字，不像旧版裁右侧避圆点）：
   - 上行 `oriented[0.30S:0.55S, 0.10S:0.90S]`（含 d0 d1；三角在 y<0.25S，已避开）
   - 下行 `oriented[0.55S:0.82S, 0.10S:0.90S]`（含 d2 校验）
3. 各 3× 放大 → RapidOCR `reader(crop, use_det=False, use_cls=False, use_rec=True)`。
4. 拼接两行文本，**保留 `0-9` 与 `X`**（`keep = [c for c in text if c in "0123456789X"]`），
   取前 4 字符。
5. 校验 `_validate(chars, conf, min_conf)`：长度≥4 且 conf≥min_conf；前 3 位是数字、
   第 4 位 ∈ `0-9`/`X`；`id = int(chars[:3])`；`chars[3] == checksum_char(id)` 才通过，
   返回 `(id, conf)`，否则 `(-1, 0)`。

## 6. `main()` 摄像头循环

照搬 `digit_detect.main` 的相机/CLAHE/曝光/trackbar/YOLO-OBB/叠加骨架，仅把定向与
识别换成本文件的。`--source`（默认 0）、`--model`（默认 `models/digi_marker_obb.pt`）、
`--conf`、`--mirror`、`--camera-exposure`。叠加：成功画绿框 + ID + 方向箭头（用
`marker_up_vector`/`id_anchor` 传 triangle_corner）；定向/校验失败画黄框 + "?"。
单帧绘制（与现有 main 一致）；`MarkerTracker` 已 import，可选接入多帧投票（默认不接，
保持与 digit_detect 一致的简单单帧）。

## 7. 测试与验证

- **定向（核心，离线、抗模糊回归）**：`generate_marker_tri(id, pixels=200)` 得正立 marker
  （三角在 TL）。对 rot ∈ {0,90,180,270}：`cv2.rotate` 后三角到已知角，断言
  `find_triangle_corner` 返回该角；`orient_by_triangle` 后再测应回到 TL(=0)。
  **加重退化**：对每个旋转再叠 `cv2.GaussianBlur(sigma≈3)` + 降对比（线性压到
  ~90–170 灰阶），断言定向仍正确——把实拍模糊钉进回归。
- **校验（纯函数）**：`_validate("0204",1.0,0.5)→(20,_)`；`_validate("007X",...)→(7,_)`；
  `_validate("0205",...)→(-1,0)`（错校验）；`_validate("007",...)→(-1,0)`（缺位）。
- **端到端（RapidOCR 装了才跑，否则 skip）**：`generate_marker_tri(id)` 小幅透视/旋转
  → `DigitRecognizerTri.recognize` → 断言读回正确 id（覆盖含 X 的，如 7/22/999）。
- **可视存档**：在用户实拍图（若提供）/合成图上跑 `main` 单帧逻辑，叠加确认 ID，存
  `docs/test-screenshots/<时间戳>/`，给路径人工核对 "?"→ID。

## 8. 范围边界（YAGNI）

本 plan 只做：定位（先复用现有 OBB）+ 三角定向 + 读 ID + 加权校验 + 叠加显示，并
**验证现有 OBB 在 tri marker 上的检出率**。

不做（留作后续/另起）：
- **重训 OBB**：仅当上面的验证显示现有 `digi_marker_obb.pt` 检出不足时才触发，届时
  另起独立 spec/plan（合成 tri 数据 + 训练 + 评估），不混进本 plan。
- 不碰 `digit_detect.py`/`digit_detect3.py`/其他解码器；不发 TUIO；不做屏幕映射/坐标归一化。


