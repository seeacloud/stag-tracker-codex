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
