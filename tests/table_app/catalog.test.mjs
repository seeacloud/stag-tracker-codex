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
