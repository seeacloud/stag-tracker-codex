# Reactable 香水桌面 Demo — 设计文档

- 日期：2026-06-04
- 阶段：A（"物理 AI 的魔法瞬间"）
- 目标：参加创业大赛 / 吸引投资的可演示 demo

## 1. 背景与目标

把现有的 STag marker 追踪链路（摄像头 → STag 识别 → TUIO → web）打磨成一个
Reactable 风格的香水桌面 demo。核心是让评委/投资人记住"物理 AI 的魔法瞬间"：

> 拿起一张印有 STag 的香水试香纸片放到桌面上，投影立刻在纸片**周围**长出
> 这款香水的信息卡片；多片共存时片与片之间长出关系连线；把两片凑近，
> 自动展开它们的详细对比。

平台愿景包含 A（魔法体验）、B（商业/数据闭环）、C（顾客记忆与体验）三层，
**本 spec 只覆盖 A**。B、C 留作后续阶段，本阶段明确不做。

### 硬件（用户已自行搭好）

经典 Reactable 背投结构：半透明桌面 + 桌下摄像头朝上拍 + 桌下投影仪背投。
软件设计为投影/坐标无关，前端保留镜像开关作为兜底。

### 用户已确认的关键决策

- 实物：香水试香纸片（blotter），每片印一个**任意** STag，一片 = 一款香水
- 摆台：Reactable 背投（硬件已就绪）
- 主视觉：信息卡片环绕纸片（轨道跟着纸片转，文字始终水平）
- 多片：各有卡片环 + 片间相似度连线（亮度 = 相似度）
- 凑近手势：把两片凑近自动展开对比，拉远收起（距离手势，零额外硬件）
- 待机态：品牌氛围光晕（暗色流体 + 若隐若现香调色 + 品牌名）
- ID 模型：ID→香水 是独立可配置映射表，任意 ID 可用，未映射 ID 优雅降级
- 渲染：DOM 为主（卡片/文字/对比），canvas 仅作底层特效（光晕 + 连线）

## 2. 架构与文件结构

### 数据流（全程复用已验证链路）

```
摄像头 → STag 识别 → TUIO/2Dobj (UDP:3333)
  → tuio_table_app.py 后端 hub（复用 SharedTuioState + SSE）
  → 浏览器 GET /events (SSE, 30Hz)
  → table_app 前端渲染器（新建）
```

### 新建文件

```
vision_fusion/
  tuio_table_app.py        # 启动器（照搬 tuio_perfume_app.py，改 ASSET_DIR + 端口）
  table_app/
    index.html             # #stage 容器 + #fx canvas + 最小 DOM
    renderer.js            # 桌面渲染器（待机/卡片环/连线/对比）— 核心
    catalog.js             # 纯香水数据（无 ID）
    id_mapping.js          # ★ ID→香水 映射表（独立可配置层）
    scent.js               # 相似度 / 对比计算
    styles.css             # 桌面样式
```

### 关键架构决策

1. **后端零改动** — `tuio_table_app.py` 完全复用 `tuio_watch_app.py` 的
   `SharedTuioState / make_handler / run_tuio_udp_listener`，只换 `ASSET_DIR`
   和端口（8778，避开 perfume 8776 / box 8787）。
2. **现有 app 原封不动** — perfume_app / box_app / watch_app 一律不碰，守住
   working state。
3. **映射层独立** — `id_mapping.js` 是 `{ stagId: catalogKey }` 表，换商家 =
   换这张表。香水数据与"哪个 ID 是哪款"彻底解耦。
4. **未映射 ID 优雅降级** — 桌上放了映射表里没有的 STag，不报错，显示
   "未注册香片 · STag #<id>" 灰色占位环，体现平台对任意 ID 的鲁棒性。
5. **渲染分层** — DOM 负责卡片/文字/对比（好排版、好动画、可访问 + CSS
   transform 实现轨道跟转/文字反转）；canvas 仅画 DOM 不擅长的两样：待机
   光晕、片间连线。

## 3. 数据模型（三层解耦）

### 第 1 层 — catalog.js（纯香水数据，无 ID）

```js
const CATALOG = {
  "nocturne-cedar": {
    name: "NOCTURNE CEDAR 02",
    house: "Maison Interface",
    family: "Woody Amber",
    price: "$168 / 75 ml",
    accent: "#38d9b3",                          // 主题色（卡片环/光晕/连线）
    hero: "Dry cedar, mineral amber...",
    notes: { top: [...], heart: [...], base: [...] },
    accords: [ { name: "Woody", value: 92 }, ... ],  // 相似度/对比依据
  },
  // ... 其余款（沿用现有 5 款数据，去掉 tagIds / similar）
};
```

key 是与 STag 无关的稳定内部标识。移除了 `tagIds`（移到映射层）和
`similar`（改为运行时由 accords 计算）。

### 第 2 层 — id_mapping.js（可配置映射表）

```js
const ID_MAPPING = {
  store: "Maison Interface · 旗舰店",   // 这张表属于哪个商家
  map: {
    64: "nocturne-cedar",
    7:  "citrus-rain",
    12: "rose-archive",
    0:  "amber-smoke",
    33: "iris-signal",
  },
};
```

任意 STag ID 可写入 `map` 指向任意 catalog key。换商家 = 替换整个文件。
未来接后端数据库时，只需让后端吐出同形状 JSON，前端不动。

### 第 3 层 — 运行时解析（renderer.js）

```js
function resolve(symbolId) {
  const key = ID_MAPPING.map[symbolId];
  if (!key) return { unregistered: true, symbolId };   // 优雅降级
  return { ...CATALOG[key], symbolId, key };
}
```

## 4. 渲染设计

### 4.1 DOM/canvas 分层结构

```
<div id="stage">                      ← 全屏容器
  <canvas id="fx">                    ← 底层：只画待机光晕 + 片间连线
  <div class="anchor" data-session>   ← 每张纸片一个，绝对定位 + rotate(angle)
     <div class="card">…</div>          ← 卡片们，反向 rotate(-angle) → 文字水平
  </div>
  ...
  <div class="compare-card">…</div>   ← 凑近对比卡（浮现于两片中点）
  <div id="status">                   ← 极简连接状态角标（默认隐藏，断线时显示"重连中"）
</div>
```

### 4.2 坐标与背投镜像

SSE 给归一化 `x/y`(0-1) → 乘 canvas 宽高 = 桌面像素位置。前端保留
`MIRROR_X` 常量作兜底（背投画面相对观看者镜像；`stag_only.py` 已处理，
此处可补可关，按硬件实测设默认）。

### 4.3 待机态（品牌氛围）

无纸片时：深色底 + 2-3 团缓慢游动的低饱和径向渐变光晕（暗金/暗青/暗紫，
sin/cos 驱动漂移）+ 中央极淡品牌字 `SCENT INTERFACE`（呼吸式透明度）。
第一张纸片落桌 → 光晕在 ~600ms 内平滑让位给卡片环（非硬切）。

### 4.4 单片卡片环（轨道跟转，文字水平）

- 一张纸片 = 一个 `.anchor`，定位到其 x/y 并 `transform: rotate(angle)`
  （angle 来自 TUIO 弧度）。
- 卡片作为 `.anchor` 子元素沿半径 R 圆周分布 → 纸片转，整圈卡片绕锚点转
  （轨道跟转）。
- 每张卡片再 `transform: rotate(-angle)` 反转 → 文字始终水平易读。
- 卡片内容拆 3-4 张绕圈排：香水名 / 香调条(accords 横条) / 前中后调 /
  价格 + hero 文案。
- 出现动画：从锚点"绽放"（半径 0→R + 淡入，~400ms，CSS transition）；
  撤走时收回淡出。

### 4.5 未注册纸片

灰色调极简卡片环："未注册香片 / STag #<id> / 等待绑定商品"。形状一致、
配色中性，证明系统认得任意标签（平台叙事素材）。

### 4.6 帧循环与平滑

`requestAnimationFrame` 持续重绘（光晕/动画需连续帧）。SSE 回调只更新
"目标状态"，渲染层对 position/angle 做轻插值(ease)，吸收识别抖动，卡片
移动如丝。

## 5. 多片关系连线 + 凑近对比

### 5.1 相似度计算（scent.js）

所有 accords 香调名汇成全集 → 每款香水投影成同维向量（缺维补 0）→
余弦相似度，落在 0-1。运行时算，不写死。
性质：`sim(a,b)==sim(b,a)`；同款=1；正交款≈0。

### 5.2 相似度连线（canvas 底层）

桌上 ≥2 片时两两画线：
- 连接两锚点（实时跟随位置）。
- 亮度/粗细 = 相似度；低于阈值(~0.3)不画，避免糊成网。
- 颜色：两端 accent 色渐变。
- 常显层：有多片即在。

### 5.3 凑近对比（距离手势）

```
两片锚点像素距离 d:
  d > FAR(180px)   → 仅细相似度线
  d < NEAR(140px)  → 展开对比卡
  之间             → 对比卡按距离淡入（连续过渡）
```

- **滞回(hysteresis)**：展开阈值(140px) ≠ 收起阈值(180px)，避免临界抖动
  导致对比卡闪烁。对 demo 稳定性关键。
- **对比卡(DOM)** 浮现于两片中点，左右分栏列关键差异。
- 差异由 scent.js 算：对比两者 accords，挑差最大的几维生成导购式文案
  （`更甜 +28` / `更木质 +44` / 价格差），箭头指向对应片。

### 5.4 多片(3+)行为

- 相似度线：所有配对（亮度过滤后通常没几条）。
- 凑近对比：只对"当前最近且 <NEAR 的一对"展开，一次一个对比卡。

### 5.5 数据兜底

未注册纸片无 accords → 不参与相似度/对比，只显示自己的灰色占位环，不污染
连线计算。

## 6. 后端启动器与错误处理

### 6.1 tuio_table_app.py

照搬 `tuio_perfume_app.py`，改三处：`ASSET_DIR` 指向 table_app、默认
`--http-port 8778`、`--demo` 合成对象复用现成逻辑。TUIO 端口仍 3333。
后端逻辑零改动。运行：`python -m vision_fusion.tuio_table_app`（项目根执行）。

### 6.2 错误处理

- SSE 断线：`EventSource.onerror` 浏览器原生自动重连，HUD 显示"重连中"。
- 未映射 ID：灰色占位环，不崩。
- 识别抖动：前端插值平滑。
- 纸片瞬时丢失：后端 stale 超时 + 前端淡出动画。
- 空/缺失映射表：所有纸片走未注册分支，仍能跑。

### 6.3 安全说明

HTTP 默认绑 `127.0.0.1`（仅本机），无认证，对本地 demo 合适。若现场需从
别的设备访问而改 `0.0.0.0`，则成为无认证的局域网开放端口 — README 注明，
默认保持本机。

## 7. 测试与验证

1. **单元测试**（scent 计算）：相似度对称性、同款=1、正交≈0、对比方向正确。
2. **--demo 模式**：合成纸片漂移 + 周期凑近，验证卡片环/连线/对比触发。
3. **Playwright 自动截图存档**：打开 `http://127.0.0.1:8778/`，截图存
   `docs/test-screenshots/<时间戳>/`：待机态 / 单片卡片环 / 多片连线 /
   凑近对比展开 — 给用户具体路径人工核查。
4. **真机验证**：用真纸片跑一遍（摄像头链路已通，stag:62 可追踪）。

## 8. 范围边界（YAGNI / 守住 A 阶段）

不做：
- B（商业/数据闭环面板）、C（顾客 LOVE/PASS 记忆）— 后续阶段。
- 不碰 perfume_app / box_app / watch_app。
- 不做后台管理界面（映射表用静态 JS 文件，够 demo）。
- 不做触摸识别（凑近用距离手势，无需触摸硬件）。
