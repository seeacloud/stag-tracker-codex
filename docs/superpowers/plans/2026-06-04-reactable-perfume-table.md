# Reactable 香水桌面 Demo (A 阶段) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建一个 Reactable 风格的香水桌面前端（`vision_fusion/table_app/`）+ 复用现有 TUIO 后端的启动器，让任意带 STag 的香水试香纸片放到桌上时，周围长出信息卡片环、多片间长出相似度连线、凑近两片自动展开对比。

**Architecture:** 后端零改动，新建 `tuio_table_app.py` 复用 `tuio_watch_app.py` 的 `SharedTuioState / make_handler / run_tuio_udp_listener`，仅换 `ASSET_DIR` 和端口 8778。前端 DOM 为主（卡片/文字/对比 + CSS transform 实现轨道跟转/文字反转），底层一张 canvas 只画待机光晕和片间连线。ID→香水 是独立可配置映射层（`id_mapping.js`），香水数据（`catalog.js`）与映射解耦，未映射 ID 优雅降级。

**Tech Stack:** Python 3 标准库（http.server，已有）、原生 JS（无框架）、HTML5 canvas + DOM、CSS transform/transition。测试：`node --test`（Node 24 内置，纯逻辑模块用 ESM 导出）、pytest（已装，测启动器 import）、Playwright MCP（已连，自动截图存档）。

---

## File Structure

**新建（前端 `vision_fusion/table_app/`）：**
- `catalog.js` — 纯香水数据（无 ID），导出 `CATALOG` 对象。同时 `export` 供 node 测试。
- `id_mapping.js` — `{ store, map: { stagId: catalogKey } }` 映射表，可配置。
- `scent.js` — 相似度/对比纯函数：`accordVector`、`similarity`、`topDifferences`。ESM `export` 供 node 测试，浏览器经全局兜底。
- `model.js` — 解析与配对纯逻辑：`resolve(symbolId)`（ID→香水，未映射降级）、`proximityState`（凑近对比的滞回状态机）。ESM `export`，可 node 测试。
- `renderer.js` — 主渲染器：SSE 接入、卡片环 DOM、canvas 光晕+连线、平滑插值。调用 model.js / scent.js，自身只管 DOM/canvas。
- `index.html` — `#stage` 容器 + `#fx` canvas + `#status` 角标 + 脚本引入。
- `styles.css` — 桌面/卡片/对比/待机样式。

**新建（后端启动器）：**
- `vision_fusion/tuio_table_app.py` — 照搬 `tuio_perfume_app.py` 结构，改 `ASSET_DIR` + 端口。

**新建（测试）：**
- `tests/table_app/scent.test.mjs` — scent.js 纯函数单测（node --test）。
- `tests/table_app/catalog.test.mjs` — catalog/mapping 数据完整性单测。
- `tests/table_app/model.test.mjs` — resolve / proximity 滞回 状态机单测。
- `tests/test_tuio_table_app.py` — 启动器可 import、ASSET_DIR/端口正确。

**不碰：** `perfume_app/` `box_app/` `watch_app/` `tuio_watch_app.py` 及其余现有文件。

---

### Task 1: scent.js 相似度/对比纯函数（TDD）

**Files:**
- Create: `vision_fusion/table_app/scent.js`
- Test: `tests/table_app/scent.test.mjs`

- [ ] **Step 1: Write the failing test**

写入 `tests/table_app/scent.test.mjs`：

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { accordVector, similarity, topDifferences } from "../../vision_fusion/table_app/scent.js";

const woody = [
  { name: "Woody", value: 92 },
  { name: "Amber", value: 72 },
  { name: "Smoke", value: 44 },
];
const citrus = [
  { name: "Citrus", value: 96 },
  { name: "Green", value: 80 },
];

test("accordVector projects onto a shared universe, missing dims = 0", () => {
  const universe = ["Woody", "Amber", "Citrus"];
  assert.deepEqual(accordVector(woody, universe), [92, 72, 0]);
  assert.deepEqual(accordVector(citrus, universe), [0, 0, 96]);
});

test("similarity of identical accords is 1", () => {
  assert.ok(Math.abs(similarity(woody, woody) - 1) < 1e-9);
});

test("similarity is symmetric", () => {
  assert.ok(Math.abs(similarity(woody, citrus) - similarity(citrus, woody)) < 1e-9);
});

test("orthogonal accords (no shared dims) similarity ~ 0", () => {
  assert.ok(similarity(woody, citrus) < 1e-9);
});

test("similarity stays within [0,1]", () => {
  const s = similarity(woody, [{ name: "Woody", value: 40 }, { name: "Citrus", value: 30 }]);
  assert.ok(s >= 0 && s <= 1);
});

test("topDifferences returns biggest-gap dims with direction toward b", () => {
  const a = [{ name: "Sweet", value: 20 }, { name: "Woody", value: 90 }];
  const b = [{ name: "Sweet", value: 70 }, { name: "Woody", value: 50 }];
  const diffs = topDifferences(a, b, 2);
  // Sweet: b-a = +50 (b 更甜), Woody: b-a = -40 (b 更不木质)
  assert.equal(diffs[0].name, "Sweet");
  assert.equal(diffs[0].delta, 50);
  assert.equal(diffs[1].name, "Woody");
  assert.equal(diffs[1].delta, -40);
});

test("topDifferences handles empty accords without throwing", () => {
  assert.deepEqual(topDifferences([], [], 3), []);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/table_app/scent.test.mjs`
Expected: FAIL — `Cannot find module .../scent.js` 或 import 报错（文件还不存在）。

- [ ] **Step 3: Write minimal implementation**

写入 `vision_fusion/table_app/scent.js`：

```js
// 香水相似度 / 对比 纯函数。无副作用，可在 node 测试与浏览器中复用。

// 把 [{name,value}] 投影到给定香调名顺序的向量，缺失维补 0。
export function accordVector(accords, universe) {
  const byName = new Map((accords || []).map((a) => [a.name, Number(a.value) || 0]));
  return universe.map((name) => byName.get(name) || 0);
}

// 两组 accords 的香调名并集（稳定顺序：先 a 后 b 的新名字）。
export function accordUniverse(a, b) {
  const seen = [];
  const add = (accords) => {
    for (const it of accords || []) if (!seen.includes(it.name)) seen.push(it.name);
  };
  add(a);
  add(b);
  return seen;
}

// 余弦相似度，落在 [0,1]（accords 值非负，故无负值）。空向量返回 0。
export function similarity(a, b) {
  const universe = accordUniverse(a, b);
  const va = accordVector(a, universe);
  const vb = accordVector(b, universe);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < universe.length; i += 1) {
    dot += va[i] * vb[i];
    na += va[i] * va[i];
    nb += vb[i] * vb[i];
  }
  if (na === 0 || nb === 0) return 0;
  const s = dot / (Math.sqrt(na) * Math.sqrt(nb));
  return Math.max(0, Math.min(1, s));
}

// 对比 a→b，按 |b-a| 降序取前 limit 个维度。delta>0 表示 b 在该维更强。
export function topDifferences(a, b, limit = 3) {
  const universe = accordUniverse(a, b);
  const va = accordVector(a, universe);
  const vb = accordVector(b, universe);
  return universe
    .map((name, i) => ({ name, delta: vb[i] - va[i] }))
    .filter((d) => d.delta !== 0)
    .sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta))
    .slice(0, limit);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/table_app/scent.test.mjs`
Expected: PASS — 7 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/table_app/scent.js tests/table_app/scent.test.mjs
git commit -m "feat(table_app): add scent similarity/diff pure functions with tests"
```

---

### Task 2: catalog.js + id_mapping.js 数据层（TDD）

**Files:**
- Create: `vision_fusion/table_app/catalog.js`
- Create: `vision_fusion/table_app/id_mapping.js`
- Test: `tests/table_app/catalog.test.mjs`

- [ ] **Step 1: Write the failing test**

写入 `tests/table_app/catalog.test.mjs`：

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { CATALOG } from "../../vision_fusion/table_app/catalog.js";
import { ID_MAPPING } from "../../vision_fusion/table_app/id_mapping.js";

test("catalog has 5 fragrances, each with required fields", () => {
  const keys = Object.keys(CATALOG);
  assert.equal(keys.length, 5);
  for (const key of keys) {
    const p = CATALOG[key];
    assert.ok(p.name, `${key} missing name`);
    assert.ok(p.accent, `${key} missing accent`);
    assert.ok(Array.isArray(p.accords) && p.accords.length > 0, `${key} bad accords`);
    assert.ok(p.notes && p.notes.top && p.notes.heart && p.notes.base, `${key} bad notes`);
    assert.ok(p.price, `${key} missing price`);
  }
});

test("catalog carries no STag id fields (ids live in mapping layer)", () => {
  for (const key of Object.keys(CATALOG)) {
    assert.equal(CATALOG[key].tagIds, undefined, `${key} should not embed tagIds`);
  }
});

test("every mapping target points to an existing catalog key", () => {
  for (const [stagId, key] of Object.entries(ID_MAPPING.map)) {
    assert.ok(CATALOG[key], `mapping ${stagId} -> ${key} has no catalog entry`);
  }
});

test("mapping declares a store label", () => {
  assert.ok(typeof ID_MAPPING.store === "string" && ID_MAPPING.store.length > 0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/table_app/catalog.test.mjs`
Expected: FAIL — 找不到 catalog.js / id_mapping.js。

- [ ] **Step 3a: Write catalog.js**

写入 `vision_fusion/table_app/catalog.js`（数据来自现有 `perfume_app/perfume_catalog.js`，去掉 `tagIds` 与 `similar`，key 为稳定 slug）：

```js
// 纯香水数据。与 STag ID 无关；ID 归属在 id_mapping.js。
export const CATALOG = {
  "nocturne-cedar": {
    name: "NOCTURNE CEDAR 02",
    house: "Maison Interface",
    family: "Woody Amber",
    concentration: "Eau de Parfum",
    price: "$168 / 75 ml",
    accent: "#38d9b3",
    hero: "Dry cedar, mineral amber, and a quiet smoke trail.",
    notes: {
      top: ["Bergamot peel", "Pink pepper", "Cold cardamom"],
      heart: ["Cedar leaf", "Iris root", "Mineral accord"],
      base: ["Virginia cedar", "Amber resin", "Soft smoke"],
    },
    accords: [
      { name: "Woody", value: 92 },
      { name: "Amber", value: 72 },
      { name: "Smoke", value: 44 },
      { name: "Fresh", value: 34 },
    ],
  },
  "citrus-rain": {
    name: "CITRUS RAIN 11",
    house: "Maison Interface",
    family: "Fresh Citrus",
    concentration: "Eau de Toilette",
    price: "$128 / 100 ml",
    accent: "#7ee35d",
    hero: "Green mandarin, wet basil, and sun-warmed white musk.",
    notes: {
      top: ["Mandarin", "Lime zest", "Rain accord"],
      heart: ["Basil", "Neroli", "Green tea"],
      base: ["White musk", "Vetiver mist", "Cedar water"],
    },
    accords: [
      { name: "Citrus", value: 96 },
      { name: "Green", value: 80 },
      { name: "Aquatic", value: 58 },
      { name: "Musk", value: 42 },
    ],
  },
  "rose-archive": {
    name: "ROSE ARCHIVE 19",
    house: "Maison Interface",
    family: "Floral Musk",
    concentration: "Eau de Parfum",
    price: "$156 / 75 ml",
    accent: "#ff6f91",
    hero: "A modern rose with pear skin, clean musk, and a lacquered petal shine.",
    notes: {
      top: ["Pear skin", "Lychee", "Sparkling aldehyde"],
      heart: ["Damask rose", "Peony", "Violet leaf"],
      base: ["Clean musk", "Cashmere wood", "Pink pepper"],
    },
    accords: [
      { name: "Rose", value: 90 },
      { name: "Musk", value: 70 },
      { name: "Fruity", value: 48 },
      { name: "Powder", value: 38 },
    ],
  },
  "amber-smoke": {
    name: "AMBER SMOKE 27",
    house: "Maison Interface",
    family: "Gourmand Resin",
    concentration: "Parfum",
    price: "$210 / 50 ml",
    accent: "#ffba55",
    hero: "Vanilla resin, dark tonka, and a polished ember finish.",
    notes: {
      top: ["Saffron", "Black pepper", "Orange bitter"],
      heart: ["Labdanum", "Tonka bean", "Burnished vanilla"],
      base: ["Amber resin", "Guaiac wood", "Smoked sugar"],
    },
    accords: [
      { name: "Amber", value: 94 },
      { name: "Sweet", value: 72 },
      { name: "Smoke", value: 62 },
      { name: "Spice", value: 48 },
    ],
  },
  "iris-signal": {
    name: "IRIS SIGNAL 33",
    house: "Maison Interface",
    family: "Powdery Floral",
    concentration: "Eau de Parfum",
    price: "$182 / 75 ml",
    accent: "#bda8ff",
    hero: "Cool iris, violet air, and a polished cosmetic-powder finish.",
    notes: {
      top: ["Violet leaf", "Aldehydes", "Lemon mist"],
      heart: ["Iris butter", "Orris root", "White suede"],
      base: ["Musk", "Cedar pencil", "Rice powder"],
    },
    accords: [
      { name: "Powder", value: 88 },
      { name: "Iris", value: 82 },
      { name: "Musk", value: 54 },
      { name: "Woody", value: 38 },
    ],
  },
};
```

- [ ] **Step 3b: Write id_mapping.js**

写入 `vision_fusion/table_app/id_mapping.js`（沿用现有 perfume_catalog 的 tagIds 关系：64→cedar, 7→citrus, 12→rose, 0→amber, 33→iris）：

```js
// ID→香水 映射表。换商家 = 替换整个文件。任意 STag ID 可指向任意 catalog key。
export const ID_MAPPING = {
  store: "Maison Interface · 旗舰店",
  map: {
    64: "nocturne-cedar",
    7: "citrus-rain",
    12: "rose-archive",
    0: "amber-smoke",
    33: "iris-signal",
  },
};
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/table_app/catalog.test.mjs`
Expected: PASS — 4 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/table_app/catalog.js vision_fusion/table_app/id_mapping.js tests/table_app/catalog.test.mjs
git commit -m "feat(table_app): add decoupled catalog + id_mapping data layer with tests"
```

---

### Task 3: tuio_table_app.py 后端启动器（TDD）

**Files:**
- Create: `vision_fusion/tuio_table_app.py`
- Test: `tests/test_tuio_table_app.py`

后端逻辑完全复用 `tuio_watch_app.py`，本任务只验证启动器配置正确（ASSET_DIR 指向 table_app、默认端口 8778、能 import 且 parse_args 可用）。不起真实服务器以保持测试快且无副作用。

- [ ] **Step 1: Write the failing test**

写入 `tests/test_tuio_table_app.py`：

```python
import sys


def test_asset_dir_points_to_table_app():
    from vision_fusion import tuio_table_app

    assert tuio_table_app.ASSET_DIR.name == "table_app"
    assert tuio_table_app.ASSET_DIR.is_dir()


def test_default_http_port_is_8778():
    from vision_fusion import tuio_table_app

    argv = sys.argv
    sys.argv = ["tuio_table_app"]
    try:
        args = tuio_table_app.parse_args()
    finally:
        sys.argv = argv
    assert args.http_port == 8778
    assert args.tuio_port == 3333


def test_reuses_watch_app_backend_symbols():
    from vision_fusion import tuio_table_app

    # 复用而非复制：这些名字应来自 tuio_watch_app
    assert hasattr(tuio_table_app, "SharedTuioState")
    assert hasattr(tuio_table_app, "make_handler")
    assert hasattr(tuio_table_app, "run_tuio_udp_listener")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tuio_table_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_fusion.tuio_table_app'`。

- [ ] **Step 3: Write minimal implementation**

写入 `vision_fusion/tuio_table_app.py`（照搬 `tuio_perfume_app.py`，改 ASSET_DIR、默认端口、analytics 默认关；保留 `--demo`）：

```python
from __future__ import annotations

import argparse
import math
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from .tuio_watch_app import (
    SharedTuioState,
    clamp01,
    make_handler,
    run_tuio_udp_listener,
)


ASSET_DIR = Path(__file__).with_name("table_app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reactable perfume table demo (phase A).")
    parser.add_argument("--tuio-host", default="0.0.0.0", help="UDP address to listen on for TUIO.")
    parser.add_argument("--tuio-port", type=int, default=3333, help="UDP port to listen on for TUIO.")
    parser.add_argument("--http-host", default="127.0.0.1", help="HTTP host for the browser app.")
    parser.add_argument("--http-port", type=int, default=8778, help="HTTP port for the browser app.")
    parser.add_argument("--event-hz", type=float, default=30.0, help="SSE update frequency.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic objects for preview.")
    parser.add_argument(
        "--demo-ids",
        nargs="+",
        type=int,
        default=[64, 7, 12],
        help="Synthetic symbol IDs to animate when --demo is enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stop = threading.Event()
    state = SharedTuioState(analytics_path=None)

    threads = [
        threading.Thread(
            target=run_tuio_udp_listener,
            args=(args.tuio_host, args.tuio_port, state, stop),
            daemon=True,
        )
    ]
    if args.demo:
        threads.append(threading.Thread(target=run_demo, args=(state, stop, args.demo_ids), daemon=True))
    for thread in threads:
        thread.start()

    handler = make_handler(state, max(args.event_hz, 1.0), asset_dir=ASSET_DIR)
    server = ThreadingHTTPServer((args.http_host, args.http_port), handler)
    print(f"TUIO table app: http://{args.http_host}:{args.http_port}/")
    print(f"Listening for /tuio/2Dobj on udp://{args.tuio_host}:{args.tuio_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


def run_demo(state: SharedTuioState, stop: threading.Event, demo_ids: list[int]) -> None:
    frame = 0
    start = time.perf_counter()
    demo_ids = demo_ids or [64]
    while not stop.is_set():
        t = time.perf_counter() - start
        objects: list[dict[str, float | int]] = []
        for index, symbol_id in enumerate(demo_ids):
            # 周期性把前两片拉近再分开，便于验证凑近对比手势。
            converge = (math.sin(t * 0.25) + 1) / 2  # 0..1
            base_x = 0.5 + (index - (len(demo_ids) - 1) / 2) * 0.22
            x = base_x + (0.5 - base_x) * converge * 0.7
            y = 0.5 + math.sin(t * 0.2 + index) * 0.06
            objects.append(
                {
                    "sessionId": index + 1,
                    "symbolId": symbol_id,
                    "x": clamp01(x),
                    "y": clamp01(y),
                    "angle": t * 0.3 + index,
                    "xVelocity": 0.0,
                    "yVelocity": 0.0,
                    "angleVelocity": 0.3,
                    "motionAccel": 0.0,
                    "rotationAccel": 0.0,
                    "lastSeen": time.time(),
                }
            )
        state.set_demo_objects(frame, objects)
        frame += 1
        time.sleep(1 / 30)


if __name__ == "__main__":
    raise SystemExit(main())
```

注意：`test_asset_dir_points_to_table_app` 要求 `ASSET_DIR.is_dir()` 为真。Task 1/2 已在 `vision_fusion/table_app/` 下创建过文件，目录已存在；若按乱序执行本任务，先 `mkdir -p vision_fusion/table_app`。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tuio_table_app.py -v`
Expected: PASS — 3 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/tuio_table_app.py tests/test_tuio_table_app.py
git commit -m "feat(table_app): add tuio_table_app launcher reusing watch backend"
```

---

### Task 4: model.js — resolve + 凑近对比滞回状态机（TDD）

**Files:**
- Create: `vision_fusion/table_app/model.js`
- Test: `tests/table_app/model.test.mjs`

把渲染器里两块"有判断逻辑、值得测"的纯函数抽出来：ID 解析（含未映射降级）和凑近对比的滞回状态机（spec 5.3：展开 <140px、收起 >180px）。

- [ ] **Step 1: Write the failing test**

写入 `tests/table_app/model.test.mjs`：

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { CATALOG } from "../../vision_fusion/table_app/catalog.js";
import { ID_MAPPING } from "../../vision_fusion/table_app/id_mapping.js";
import { makeResolver, proximityState, NEAR, FAR } from "../../vision_fusion/table_app/model.js";

const resolve = makeResolver(CATALOG, ID_MAPPING);

test("resolve maps a known STag id to its catalog entry", () => {
  const r = resolve(64);
  assert.equal(r.unregistered, undefined);
  assert.equal(r.name, "NOCTURNE CEDAR 02");
  assert.equal(r.key, "nocturne-cedar");
  assert.equal(r.symbolId, 64);
});

test("resolve degrades gracefully for an unmapped id", () => {
  const r = resolve(999);
  assert.equal(r.unregistered, true);
  assert.equal(r.symbolId, 999);
  assert.equal(r.name, undefined);
});

test("NEAR < FAR so hysteresis has a dead band", () => {
  assert.ok(NEAR < FAR);
});

test("proximityState opens when distance drops below NEAR", () => {
  const next = proximityState(false, NEAR - 1);
  assert.equal(next, true);
});

test("proximityState stays open in the dead band (NEAR..FAR)", () => {
  const mid = (NEAR + FAR) / 2;
  assert.equal(proximityState(true, mid), true);   // was open -> stays open
  assert.equal(proximityState(false, mid), false); // was closed -> stays closed
});

test("proximityState closes when distance exceeds FAR", () => {
  assert.equal(proximityState(true, FAR + 1), false);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/table_app/model.test.mjs`
Expected: FAIL — 找不到 model.js。

- [ ] **Step 3: Write minimal implementation**

写入 `vision_fusion/table_app/model.js`：

```js
// 解析与配对的纯逻辑。无 DOM 依赖，可 node 测试。

// 凑近对比的像素阈值（spec 5.3）。NEAR<FAR 形成滞回死区。
export const NEAR = 140;
export const FAR = 180;

// 用给定 catalog + mapping 造一个 resolver：symbolId -> 香水（或未映射降级对象）。
export function makeResolver(catalog, mapping) {
  return function resolve(symbolId) {
    const key = mapping.map[symbolId];
    if (!key || !catalog[key]) return { unregistered: true, symbolId };
    return { ...catalog[key], symbolId, key };
  };
}

// 滞回状态机：根据上一帧是否展开 + 当前距离，决定这一帧是否展开对比卡。
// d<NEAR 开；d>FAR 关；之间维持原状。
export function proximityState(wasOpen, distance) {
  if (distance < NEAR) return true;
  if (distance > FAR) return false;
  return wasOpen;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/table_app/model.test.mjs`
Expected: PASS — 6 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/table_app/model.js tests/table_app/model.test.mjs
git commit -m "feat(table_app): add resolve + proximity hysteresis model with tests"
```

---

### Task 5: index.html + styles.css 静态外壳

**Files:**
- Create: `vision_fusion/table_app/index.html`
- Create: `vision_fusion/table_app/styles.css`

前端外壳无法用 node 单测，本任务只产出静态文件；可见行为在 Task 7 用 Playwright 截图验证。

- [ ] **Step 1: Write index.html**

写入 `vision_fusion/table_app/index.html`：

```html
<!doctype html>
<html lang="zh">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Scent Interface · Table</title>
    <link rel="stylesheet" href="/styles.css?v=20260604" />
  </head>
  <body>
    <div id="stage" aria-label="Scent Interface Table">
      <canvas id="fx"></canvas>
      <div id="layers" aria-live="polite"></div>
      <div id="compare" class="compare-card is-hidden" role="status"></div>
      <div id="standby" class="standby">
        <div class="brand">SCENT INTERFACE</div>
      </div>
      <div id="status" class="status is-hidden">重连中</div>
    </div>

    <script type="module" src="/renderer.js?v=20260604"></script>
  </body>
</html>
```

注意：`renderer.js` 用 `type="module"`，可直接 `import` catalog/mapping/scent/model。

- [ ] **Step 2: Write styles.css**

写入 `vision_fusion/table_app/styles.css`：

```css
:root {
  --bg: #07090c;
  --ink: #f4f7f5;
  --muted: rgba(244, 247, 245, 0.55);
  --card-bg: rgba(16, 22, 26, 0.86);
  --card-border: rgba(255, 255, 255, 0.1);
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
  font-family: "Inter", system-ui, "Segoe UI", sans-serif; overflow: hidden; }
#stage { position: fixed; inset: 0; }
#fx { position: absolute; inset: 0; width: 100%; height: 100%; }
#layers { position: absolute; inset: 0; pointer-events: none; }

/* 待机品牌氛围 */
.standby { position: absolute; inset: 0; display: grid; place-items: center;
  transition: opacity 600ms ease; }
.standby.is-dim { opacity: 0; }
.standby .brand { font-size: clamp(28px, 6vw, 72px); letter-spacing: 0.4em;
  font-weight: 300; color: var(--muted); animation: breathe 5s ease-in-out infinite; }
@keyframes breathe { 0%,100% { opacity: 0.25; } 50% { opacity: 0.6; } }

/* 纸片锚点 + 卡片环 */
.anchor { position: absolute; transform-origin: center; will-change: transform; }
.card { position: absolute; min-width: 160px; max-width: 220px; padding: 12px 14px;
  background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px;
  backdrop-filter: blur(8px); transform-origin: center;
  transition: opacity 400ms ease, transform 400ms ease; }
.card.is-enter { opacity: 0; }
.card h3 { margin: 0 0 4px; font-size: 15px; letter-spacing: 0.04em; }
.card .meta { font-size: 11px; color: var(--muted); }
.card .accord-row { display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 11px; }
.card .accord-bar { height: 4px; border-radius: 3px; flex: 1; }
.card.unregistered { filter: grayscale(1); opacity: 0.7; }

/* 凑近对比卡 */
.compare-card { position: absolute; transform: translate(-50%, -50%);
  display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; padding: 14px 18px;
  background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px;
  backdrop-filter: blur(10px); transition: opacity 300ms ease; max-width: 360px; }
.compare-card.is-hidden { opacity: 0; pointer-events: none; }
.compare-card .diff { font-size: 12px; }
.compare-card .vs { color: var(--muted); align-self: center; }

.status { position: absolute; top: 14px; right: 16px; font-size: 12px;
  color: #ffba55; letter-spacing: 0.1em; }
.is-hidden { display: none; }
```

- [ ] **Step 3: Verify files exist (no test framework for static HTML)**

Run: `ls vision_fusion/table_app/index.html vision_fusion/table_app/styles.css`
Expected: 两个路径都列出。

- [ ] **Step 4: Commit**

```bash
git add vision_fusion/table_app/index.html vision_fusion/table_app/styles.css
git commit -m "feat(table_app): add static shell (index.html + styles.css)"
```

---

### Task 6: renderer.js — SSE + 卡片环 + 光晕 + 连线 + 凑近对比

**Files:**
- Create: `vision_fusion/table_app/renderer.js`

整合前面所有模块。无 node 单测（依赖 DOM/canvas/SSE）；行为在 Task 7 用 Playwright 截图 + demo 模式验证。代码较长，按职责分段，全部写入同一文件。

- [ ] **Step 1: 写入 renderer.js（imports + 全局状态 + resize + SSE 接入）**

写入 `vision_fusion/table_app/renderer.js`：

```js
import { CATALOG } from "./catalog.js";
import { ID_MAPPING } from "./id_mapping.js";
import { similarity, topDifferences } from "./scent.js";
import { makeResolver, proximityState, NEAR, FAR } from "./model.js";

const resolve = makeResolver(CATALOG, ID_MAPPING);

// 背投镜像兜底：硬件已镜像则保持 false（按实测调）。
const MIRROR_X = false;
const RING_RADIUS = 140;     // 卡片环半径(px)
const CARD_SLOTS = 4;        // 一圈最多排几张卡片
const EASE = 0.18;           // 位置/角度插值系数
const SIM_LINE_MIN = 0.3;    // 相似度低于此不画连线

const stage = document.getElementById("stage");
const fx = document.getElementById("fx");
const fxCtx = fx.getContext("2d");
const layers = document.getElementById("layers");
const standby = document.getElementById("standby");
const compareEl = document.getElementById("compare");
const statusEl = document.getElementById("status");

let W = 0;
let H = 0;
let dpr = 1;
// 渲染态：sessionId -> { symbolId, info, x, y, angle, tx, ty, tAngle, el, cards }
const tracks = new Map();
let snapshot = { objects: [] };
let compareOpen = false;     // 凑近对比滞回状态

function resize() {
  dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  W = window.innerWidth;
  H = window.innerHeight;
  fx.width = Math.floor(W * dpr);
  fx.height = Math.floor(H * dpr);
  fx.style.width = `${W}px`;
  fx.style.height = `${H}px`;
  fxCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);
resize();

const source = new EventSource("/events");
source.addEventListener("tuio", (event) => {
  snapshot = JSON.parse(event.data);
  statusEl.classList.add("is-hidden");
  syncTracks(snapshot.objects || []);
});
source.onerror = () => {
  statusEl.classList.remove("is-hidden");
};
```

- [ ] **Step 2: 追加 syncTracks（增删轨道 + 设目标位置/角度）**

在 renderer.js 末尾追加：

```js
function normX(x) {
  return (MIRROR_X ? 1 - x : x) * W;
}

function syncTracks(objects) {
  const seen = new Set();
  for (const obj of objects) {
    const id = String(obj.sessionId);
    seen.add(id);
    const tx = normX(Number(obj.x));
    const ty = Number(obj.y) * H;
    const tAngle = Number(obj.angle) || 0;
    let track = tracks.get(id);
    if (!track || track.symbolId !== Number(obj.symbolId)) {
      if (track) removeTrack(id);
      track = createTrack(id, Number(obj.symbolId), tx, ty, tAngle);
      tracks.set(id, track);
    }
    track.tx = tx;
    track.ty = ty;
    track.tAngle = tAngle;
    track.lastSeen = performance.now();
  }
  for (const id of [...tracks.keys()]) {
    if (!seen.has(id)) removeTrack(id);
  }
  standby.classList.toggle("is-dim", tracks.size > 0);
}

function createTrack(id, symbolId, x, y, angle) {
  const info = resolve(symbolId);
  const el = document.createElement("div");
  el.className = "anchor";
  const cards = buildCards(info);
  for (const c of cards) el.appendChild(c);
  layers.appendChild(el);
  // 下一帧去掉 is-enter 触发绽放动画
  requestAnimationFrame(() => {
    for (const c of cards) c.classList.remove("is-enter");
  });
  return { symbolId, info, x, y, angle, tx: x, ty: y, tAngle: angle, el, cards };
}

function removeTrack(id) {
  const track = tracks.get(id);
  if (!track) return;
  track.el.remove();
  tracks.delete(id);
}
```

- [ ] **Step 3: 追加 buildCards（卡片环 DOM 内容）**

在 renderer.js 末尾追加：

```js
function buildCards(info) {
  if (info.unregistered) {
    const card = cardEl("unregistered");
    card.innerHTML = `<h3>未注册香片</h3><div class="meta">STag #${info.symbolId} · 等待绑定商品</div>`;
    return [card];
  }
  const nameCard = cardEl();
  nameCard.style.borderColor = info.accent;
  nameCard.innerHTML = `<h3>${info.name}</h3><div class="meta">${info.family} · ${info.price}</div>`;

  const accordCard = cardEl();
  accordCard.innerHTML = info.accords
    .map(
      (a) =>
        `<div class="accord-row"><span>${a.name}</span>` +
        `<span class="accord-bar" style="width:${a.value}%;background:${info.accent}"></span></div>`
    )
    .join("");

  const notesCard = cardEl();
  notesCard.innerHTML =
    `<div class="meta">前调 ${info.notes.top.join("、")}</div>` +
    `<div class="meta">中调 ${info.notes.heart.join("、")}</div>` +
    `<div class="meta">后调 ${info.notes.base.join("、")}</div>`;

  const storyCard = cardEl();
  storyCard.innerHTML = `<div class="meta">${info.hero}</div>`;

  return [nameCard, accordCard, notesCard, storyCard];
}

function cardEl(extra = "") {
  const el = document.createElement("div");
  el.className = `card is-enter ${extra}`.trim();
  return el;
}
```

- [ ] **Step 4: 追加 frame 循环（插值 + 卡片环布局 + canvas 光晕/连线 + 对比）**

在 renderer.js 末尾追加：

```js
let bgT = 0;

function frame() {
  bgT += 0.005;
  // 插值靠近目标
  for (const track of tracks.values()) {
    track.x += (track.tx - track.x) * EASE;
    track.y += (track.ty - track.y) * EASE;
    track.angle += angleDelta(track.angle, track.tAngle) * EASE;
  }
  drawBackground();
  drawLinks();
  layoutCards();
  updateCompare();
  requestAnimationFrame(frame);
}

function angleDelta(from, to) {
  let d = to - from;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function layoutCards() {
  for (const track of tracks.values()) {
    // 锚点定位到纸片，整体旋转 = 轨道跟转
    track.el.style.left = `${track.x}px`;
    track.el.style.top = `${track.y}px`;
    track.el.style.transform = `rotate(${track.angle}rad)`;
    const n = track.cards.length;
    track.cards.forEach((card, i) => {
      const theta = (i / Math.max(1, n)) * Math.PI * 2 - Math.PI / 2;
      const cx = Math.cos(theta) * RING_RADIUS;
      const cy = Math.sin(theta) * RING_RADIUS;
      // 卡片位置随环走；自身反向旋转 -> 文字始终水平
      card.style.left = `${cx}px`;
      card.style.top = `${cy}px`;
      card.style.transform = `translate(-50%, -50%) rotate(${-track.angle}rad)`;
    });
  }
}

function drawBackground() {
  fxCtx.clearRect(0, 0, W, H);
  fxCtx.fillStyle = "#07090c";
  fxCtx.fillRect(0, 0, W, H);
  const blobs = [
    { c: "#1d3a33", ox: 0.3, oy: 0.4, r: 0.5 },
    { c: "#2a2742", ox: 0.7, oy: 0.6, r: 0.45 },
    { c: "#3a2e1a", ox: 0.5, oy: 0.3, r: 0.4 },
  ];
  for (let i = 0; i < blobs.length; i += 1) {
    const b = blobs[i];
    const x = (b.ox + Math.sin(bgT + i) * 0.05) * W;
    const y = (b.oy + Math.cos(bgT * 0.8 + i) * 0.05) * H;
    const rad = b.r * Math.min(W, H);
    const g = fxCtx.createRadialGradient(x, y, 0, x, y, rad);
    g.addColorStop(0, b.c);
    g.addColorStop(1, "rgba(7,9,12,0)");
    fxCtx.fillStyle = g;
    fxCtx.fillRect(0, 0, W, H);
  }
}
```

- [ ] **Step 5: 追加 drawLinks + updateCompare（连线 + 凑近对比）**

在 renderer.js 末尾追加：

```js
function activeRegistered() {
  return [...tracks.values()].filter((t) => !t.info.unregistered);
}

function drawLinks() {
  const items = activeRegistered();
  for (let i = 0; i < items.length; i += 1) {
    for (let j = i + 1; j < items.length; j += 1) {
      const a = items[i];
      const b = items[j];
      const sim = similarity(a.info.accords, b.info.accords);
      if (sim < SIM_LINE_MIN) continue;
      const grad = fxCtx.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, a.info.accent);
      grad.addColorStop(1, b.info.accent);
      fxCtx.strokeStyle = grad;
      fxCtx.globalAlpha = sim;
      fxCtx.lineWidth = 1 + sim * 5;
      fxCtx.beginPath();
      fxCtx.moveTo(a.x, a.y);
      fxCtx.lineTo(b.x, b.y);
      fxCtx.stroke();
    }
  }
  fxCtx.globalAlpha = 1;
}

function closestPair(items) {
  let best = null;
  for (let i = 0; i < items.length; i += 1) {
    for (let j = i + 1; j < items.length; j += 1) {
      const dx = items[i].x - items[j].x;
      const dy = items[i].y - items[j].y;
      const d = Math.hypot(dx, dy);
      if (!best || d < best.d) best = { a: items[i], b: items[j], d };
    }
  }
  return best;
}

function updateCompare() {
  const items = activeRegistered();
  const pair = items.length >= 2 ? closestPair(items) : null;
  compareOpen = pair ? proximityState(compareOpen, pair.d) : false;
  if (!pair || !compareOpen) {
    compareEl.classList.add("is-hidden");
    return;
  }
  const { a, b } = pair;
  const diffs = topDifferences(a.info.accords, b.info.accords, 3);
  const dir = (d) => (d.delta > 0 ? `${b.info.name} 更${d.name} +${d.delta}` : `${a.info.name} 更${d.name} +${-d.delta}`);
  compareEl.innerHTML =
    `<div class="diff"><strong>${a.info.name}</strong></div>` +
    `<div class="vs">VS</div>` +
    `<div class="diff"><strong>${b.info.name}</strong></div>` +
    `<div class="diff" style="grid-column:1/-1">` +
    diffs.map((d) => `<div>${dir(d)}</div>`).join("") +
    `</div>`;
  compareEl.style.left = `${(a.x + b.x) / 2}px`;
  compareEl.style.top = `${(a.y + b.y) / 2}px`;
  compareEl.classList.remove("is-hidden");
}

requestAnimationFrame(frame);
```

- [ ] **Step 6: 启动 demo 模式手动冒烟**

Run: `python -m vision_fusion.tuio_table_app --demo`
然后浏览器开 `http://127.0.0.1:8778/`。
Expected: 看到待机光晕淡出，3 个卡片环漂移，卡片绕纸片转、文字水平，片间有连线，前两片周期性靠近时中间浮现对比卡。`Ctrl+C` 停。

- [ ] **Step 7: Commit**

```bash
git add vision_fusion/table_app/renderer.js
git commit -m "feat(table_app): add reactable renderer (rings, links, proximity compare)"
```

---

### Task 7: 全量测试 + Playwright 截图存档 + README

**Files:**
- Create: `docs/test-screenshots/<timestamp>/` (截图产物)
- Modify: `README.md`（追加运行说明）

- [ ] **Step 1: 跑全部 JS 单测**

Run: `node --test tests/table_app/`
Expected: scent(7) + catalog(4) + model(6) 全 PASS，共 17 个。

- [ ] **Step 2: 跑 Python 启动器测试**

Run: `python -m pytest tests/test_tuio_table_app.py -v`
Expected: 3 个 PASS。

- [ ] **Step 3: 启动 demo 服务（后台）供截图**

Run: `python -m vision_fusion.tuio_table_app --demo &`
等待约 1 秒让服务起来。确认监听：`curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8778/` 应输出 `200`。

- [ ] **Step 4: 用 Playwright MCP 截图存档**

用 Playwright MCP（已连）：
1. `browser_navigate` → `http://127.0.0.1:8778/`，设视口 1280x800。
2. 等 ~1.5s（待机光晕淡出 + 卡片绽放）。
3. `browser_take_screenshot` 存到 `docs/test-screenshots/<YYYYMMDD-HHMMSS>/01-rings-and-links.png`。
4. 再等 ~4s（demo 让前两片靠近触发对比卡），再截一张 `02-proximity-compare.png`。
5. 关页面，`kill` 掉后台 demo 进程。

每张截图人工核查点：01 有多个卡片环 + 片间连线；02 中间出现对比卡。把目录路径报给用户。

- [ ] **Step 5: 更新 README**

在 `README.md` 现有 app 运行说明附近，追加（保持与现有 perfume/watch app 描述同样的风格）：

````markdown
### Reactable 香水桌面 (table_app)

从项目根运行：

```bash
python -m vision_fusion.tuio_table_app          # 接收真实 TUIO (UDP:3333)
python -m vision_fusion.tuio_table_app --demo   # 合成数据预览
```

浏览器打开 http://127.0.0.1:8778/ 。任意带 STag 的香水试香纸片放到桌面，
周围长出信息卡片环；多片间按香调相似度长出连线；把两片凑近自动展开对比。
ID→香水 映射在 `vision_fusion/table_app/id_mapping.js`，换商家改这一个文件。

> 默认仅绑 127.0.0.1（本机）。若需局域网设备访问，加 `--http-host 0.0.0.0`，
> 注意这会开放一个无认证端口。
````

- [ ] **Step 6: Commit**

```bash
git add README.md docs/test-screenshots/
git commit -m "docs(table_app): add run instructions and verification screenshots"
```

---

## 验证清单（spec 覆盖自检）

- spec §2 后端复用零改动 → Task 3
- spec §3 三层数据解耦 (catalog/mapping/resolve) → Task 2 + Task 4
- spec §4.1 DOM/canvas 分层 → Task 5 + Task 6
- spec §4.2 镜像兜底 MIRROR_X → Task 6 Step 1
- spec §4.3 待机品牌光晕 → Task 5 (standby) + Task 6 (drawBackground)
- spec §4.4 卡片环轨道跟转/文字水平 → Task 6 Step 3+4 (layoutCards 反向旋转)
- spec §4.5 未注册降级 → Task 4 (resolve) + Task 6 (buildCards unregistered)
- spec §4.6 插值平滑 → Task 6 Step 4 (EASE)
- spec §5.1 相似度计算 → Task 1
- spec §5.2 相似度连线 → Task 6 Step 5 (drawLinks)
- spec §5.3 凑近对比 + 滞回 → Task 4 (proximityState) + Task 6 Step 5 (updateCompare)
- spec §5.4 多片只展开最近一对 → Task 6 Step 5 (closestPair)
- spec §5.5 未注册不参与连线 → Task 6 Step 5 (activeRegistered 过滤)
- spec §6 启动器 + 错误处理 → Task 3 + Task 6 (SSE onerror status)
- spec §6.3 安全说明 → Task 7 (README note)
- spec §7 测试与截图 → Task 1/2/4 单测 + Task 7 Playwright






