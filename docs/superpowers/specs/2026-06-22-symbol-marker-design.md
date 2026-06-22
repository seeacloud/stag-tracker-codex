# 符号 Marker 系统设计 (symbol-marker)

**日期:** 2026-06-22
**分支:** `feat/symbol-marker`
**状态:** 设计已确认,待写实现计划

## 目标

把 tri marker 的 2×2 **数字**(0-9 + 校验 X)替换成一套 **11 个高区分度几何符号**,从源头压低识别误判;**定向标记从"左上内角黑三角"改为"底边加宽黑条"**(信号更强、抗模糊更好、不占内部、不蹭符号格)。沿用 tri 管线主体(YOLO-OBB 框定 + warp + 切格 + CNN 分类 + 加权 mod11 校验 + 多帧投票),换"每格内容"+"定向方式"。

> **定向变更(底边黑条)**:外框为完整方框,**底边比其它三边明显加粗**(一条宽黑带)。定向 = 比 4 条边的暗度(去线性平面后),最暗边 = 底边 → 据此把 marker 旋正。取代原"4 角三角暗度"。理由:整条边信号量远大于小三角,重模糊/梯度下更稳;且在外框上,不侵占内部、切格无需抹除三角。前 3 格编 ID、第 4 格加权 mod11+X 校验(不变)。

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

**复用(不改)**:YOLO-OBB 检测(`tri_marker_obb.pt`)、`warp_square`、`slice_cells` 的归一化+切格几何、`DigitCNN` 架构、加权 mod11+X 校验、`MarkerTracker` 多帧投票、解码缓存、HUD/调参。

**替换两处**:① 2×2 格子内容(数字字形 → 符号画法);② **定向方式(内角三角 → 底边黑条 + 边暗度定向)**——`slice_cells` 的"抹三角"改成不需要(底边黑条在外框、不进格);新增"4 边暗度定向"取代 `corner_ranking` 的角定向。

## 组件与改动

1. **`vision_fusion/symbol_set.py`(新建)** — 符号库的唯一真相源:
   - `SYMBOLS = "0123456789X"`(逻辑标签,沿用,方便复用 checksum_char/decode_id)
   - `draw_symbol(char, size, stroke, round_cap=True) -> np.ndarray`:按 v8 符号表画 64×64,统一线宽 stroke、圆角线头。

2. **`vision_fusion/symbol_marker.py`(新建)** — `render_marker_symbol(marker_id, **layout)`:复刻 `generate_marker_tri` 的外框 + 2×2 中心布局,但**定向用底边加宽黑条**(底边 border 比其它三边粗 `bottom_extra`),格内用 `draw_symbol`。CLI 批量导出。复用 `checksum_char`。

3. **定向(底边黑条)** — 在 `digit_detect_tri.py` 新增 `edge_ranking(square)`:对 warp 后方图,去线性平面后比 4 条边(上/右/下/左)带状区暗度,返回最暗边;`orient_by_edge` 把最暗边旋到底部。取代符号路径的 `corner_ranking`。`slice_cells` 加开关跳过"抹三角"(符号路径不需要)。

4. **训练数据 `vision_fusion/nn_synth_symbol.py`(新建)** — 渲染符号 marker → `slice_cells`(不抹三角) → 退化 + `--scene` → `datasets/symbol_crops/<char>/`。复用 `nn_augment`/`overlay_marker`。

5. **训练** — 复用 `nn_train_tri_digit`,`--data datasets/symbol_crops --out models/symbol_cnn.pt`(`DigitCNN` 11 类)。

6. **识别** — `DigitClassifierTri(model_path="models/symbol_cnn.pt")`,定向注入 `edge_ranking`;`decode_id`/`checksum_char` 不变。`digit_detect_tri.main` 加 `--symbol` 开关(切到符号模型 + 边定向 + 不抹三角)。

7. **`vision_fusion/symbol_marker_ui.py`(新建)** — Tkinter 调参 GUI(复刻 `digit_marker_tri_ui`):
   实时预览 + 区间批量导出 + 参数自动存取 `symbol_marker_settings.json`(启动加载、改动/关闭保存)。可调:**统一线宽、圆角开关、符号大小、列距、行距、padding、边框宽度、底边黑条额外宽度**。
   `draw_symbol`/`render_marker_symbol` 接受这些参数,GUI 只是前端;训练生成与识别都读同一 json,保证渲染/训练/推理一致。

## 数据流

```
帧 → 灰度 → CLAHE ─┬─ YOLO-OBB 框定
                   └─ warp 拉正 → 底边黑条定向(edge_ranking 去平面) → slice_cells(归一化+切4格,不抹三角)
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
