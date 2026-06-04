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
