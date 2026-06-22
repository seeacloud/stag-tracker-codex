# 符号 Marker 系统设计 (symbol-marker)

**日期:** 2026-06-22
**分支:** `feat/symbol-marker`
**状态:** 设计已确认,待写实现计划

## 目标

把 tri marker 的 2×2 **数字**(0-9 + 校验 X)替换成一套 **11 个高区分度几何符号**,从源头压低识别误判;沿用现有 tri 管线(YOLO-OBB 框定 + 内角黑三角定向 + warp + 切格 + CNN 分类 + 加权 mod11 校验 + 多帧投票),只换"每格内容"。

## 动机(数据支撑)

- 现用 Gothic 数字:抗模糊后**最小对距离 23.5**,最易混对是 `3/8`、`5/8` 等闭环数字。
- 换字体此路不通:实测 Gothic 已是常见字体里最优,其它(Consolas/Cascadia/Arial)更差;数字 `8` 类闭环混淆是阿拉伯数字固有缺陷。
- 选定的 11 符号集:**最小对距离 53.5(数字的 2.3 倍)**,蒙特卡洛最近邻**误判率 0.00%**(0/3300)。
- 诚实说明:三套候选(数字23.5 / v6=51 / v8=53.5)在当前退化强度下误判率**都是 0%**;符号系统的实际价值是"更大规模/更差画质下的鲁棒余量",当前 demo 的眼前问题(单张 0935 抖动)其实是**印刷缺陷**,重印即可。用户选择上符号系统作为长期升级。

## 符号集(已锁定,11 个)

索引→符号(语义沿用:前 3 格 = 数据位编码 ID,第 4 格 = 加权 mod11+X 校验):

| idx | 名称 | 画法(64×64,粗笔画 t=12) |
|---|---|---|
| 0 | dring 实心圆 | `circle((32,32),22,fill)` |
| 1 | L | 竖条(16,8,30,56)+ 底横(16,42,52,56) |
| 2 | vbar 竖条 | `rect(25,6,39,58)` |
| 3 | bslash ╲ | `line((12,12),(52,52),t)` |
| 4 | Tl ⊢ | 左竖(10,8,24,56)+ 中横(10,25,54,39) |
| 5 | corner ⌐ | 顶横(12,8,48,22)+ 右竖(34,8,48,40) |
| 6 | Y | 中竖(32,32)-(32,56) + 两斜(32,34)-(10,10)/(54,10) |
| 7 | J 反L | 右竖(34,8,48,56)+ 底横(12,42,48,56) |
| 8 | Tr ⊣ | 右竖(40,8,54,56)+ 中横(10,25,54,39) |
| 9 | fslash ╱ | `line((52,12),(12,52),t)` |
| X | box □ 空心方框 | `rect(12,12,52,52,t)` |

> 镜像对(⊢/⊣、╱/╲)经测距离最大(60+)——因三角定向已把 marker 转正、方向固定,镜像不会混。
> 最难对 dring/⊢ = 53.5。无实心糊块隐患。

> **绘制要求(贯穿所有符号)**:统一线宽(一个 stroke 值控全部线条/块边)、所有线条末端**圆角**
> (画线用圆端点 cv2.LINE_AA + round cap;实心块可选圆角)、符号大小可调。上表坐标是 t=12 时的
> 参考形状,实际由参数驱动缩放;dring 实心圆/box 空心方框的"线宽"指描边宽度(box)或随大小缩放(dring)。

## 架构:复用 + 替换

**全部复用(不改)**:YOLO-OBB 检测(`tri_marker_obb.pt`)、内角黑三角定向(`corner_ranking` 去平面)、`warp_square`、`slice_cells`(切格几何 + 归一化 + 抹三角)、`DigitCNN` 架构、加权 mod11+X 校验逻辑、`MarkerTracker` 多帧投票、解码缓存、HUD/调参。

**只替换**:2×2 格子里的字形渲染(数字字形 → 符号画法)。

## 组件与改动

1. **`vision_fusion/symbol_set.py`(新建)** — 符号库的唯一真相源:
   - `SYMBOLS = "0123456789X"`(逻辑标签,沿用,方便复用 checksum_char/decode_id)
   - `draw_symbol(idx_or_char, size) -> np.ndarray`:按上表画 64×64 符号。
   - `render_marker_symbol(marker_id, **layout)`:复刻 `generate_marker_tri` 的外框+内角黑三角+2×2 布局,但每格用 `draw_symbol` 画符号(而非数字字形)。

2. **`vision_fusion/nn_synth_tri_digit.py`(参数化或新建 symbol 版)** — 训练数据生成:marker 渲染换成符号版,切格(`slice_cells`)、退化、`--scene` 采集真实样本全沿用。输出 `datasets/symbol_crops/<char>/`。

3. **训练** — 复用 `nn_train_tri_digit`(`DigitCNN` 11 类),数据指向符号 crops,产出 `models/symbol_cnn.pt`。

4. **识别** — `DigitClassifierTri` 加 `model_path` 参数即可指向 `symbol_cnn.pt`;`decode_id`/`checksum_char` 完全不变(逻辑标签仍是 0-9+X)。

5. **`digit_detect_tri.py`** — 加 `--symbol` 开关:用 `symbol_cnn.pt` + 符号渲染。HUD 显示时把逻辑 id 映射回符号或仍显示数字 id(数字 id 更有用,符号只是物理编码)。

6. **`vision_fusion/symbol_marker_ui.py`(新建)** — Tkinter 调参 GUI(复刻 `digit_marker_tri_ui` 机制):
   实时预览 + 区间批量导出 + 参数自动存取。可调参数(滑块/输入):
   - **线宽(stroke width)**:所有符号统一用此线宽绘制(一个值控全部)。
   - **圆角线头**:所有线条末端圆角(画线用圆端点 + 抗锯齿;实心块四角可选圆角半径)。
   - **符号大小**:符号在单格内的占比。
   - **列距(字符间距)**、**行距(行高)**:2×2 格的横/纵间隔。
   - **marker padding**:内容区四周留白。
   - 三角定向标记大小、边框宽度(沿用 tri 的几何参数)。
   - **参数自动保存/加载**:存 `symbol_marker_settings.json`,启动自动加载上次值,改动/关闭时自动保存(WM_DELETE + Save 按钮,同 digit_marker_tri_ui)。

   `draw_symbol` 必须接受这些参数(线宽、圆角、大小),`render_marker_symbol` 接受布局参数(列距/行距/padding/三角大小/边框),GUI 只是它们的可视化前端。训练数据生成与实时识别都从 `symbol_marker_settings.json` 读同一套参数,保证渲染/训练/推理一致。

## 数据流(不变)

```
帧 → 灰度 → CLAHE ─┬─ YOLO-OBB 框定
                   └─ warp 拉正 → 三角定向 → slice_cells(归一化+切4格)
                       → CNN(symbol_cnn) 读 4 符号 → 逻辑标签 0-9+X
                       → decode_id 加权 mod11 校验 → marker_id → 多帧投票
```

## 测试

- `symbol_set` 单元测试:11 符号渲染尺寸/无重复/最小对距离 > 数字基线。
- 数据生成冒烟 + 切格存档(人工核对符号居中)。
- 训练后:符号 marker 端到端识别(渲染 marker → 读回 id),含校验 X。
- A/B 闸门:符号 CNN 在合成淡+梯度上 误读=0、正确率不低于数字版。
- 实拍验证 + FPS 不回退。

## 风险/权衡

- **人不可读**:符号需查表。对工业 marker 可接受(机器可读 > 人可读)。
- **要重印全部实物 marker**:大改动,但用户已确认走长期升级路线。
- **收益边际**:当前误判率已 0,符号系统是鲁棒余量;价值在更大规模/更差画质。
- 兼容:数字版 marker 与模型保留(`--recognizer`/`--symbol` 开关并存),不破坏现有 tri 数字流程。
