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
