import { test } from "node:test";
import assert from "node:assert/strict";
import { checkFile, formatPercent, prettyLabel } from "./validate.ts";

const f = (name: string, type: string, size: number) => ({ name, type, size });

test("accepts png and jpg", () => {
  assert.equal(checkFile(f("a.png", "image/png", 1000), 5).ok, true);
  assert.equal(checkFile(f("a.JPEG", "image/jpeg", 1000), 5).ok, true);
});

test("rejects empty, wrong type, too large", () => {
  assert.equal(checkFile(f("a.png", "image/png", 0), 5).ok, false);
  assert.equal(checkFile(f("a.gif", "image/gif", 10), 5).ok, false);
  assert.equal(checkFile(f("a.txt", "text/plain", 10), 5).ok, false);
  assert.equal(checkFile(f("a.png", "image/png", 6 * 1024 * 1024), 5).ok, false);
});

test("formatting helpers", () => {
  assert.equal(formatPercent(0.9312), "93.1%");
  assert.equal(formatPercent(null), "–");
  assert.equal(prettyLabel("INR_500"), "₹500");
  assert.equal(prettyLabel(null), "–");
});
