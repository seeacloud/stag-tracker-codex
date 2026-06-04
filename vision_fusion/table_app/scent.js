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
