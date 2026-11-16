import assert from 'node:assert/strict';
import test from 'node:test';
import { RenderBudget, probeSchedule } from '../web/performance.js';

function sample(budget, milliseconds, from, count = 60) {
  for (let i = 1; i <= count; i++) budget.observe(milliseconds, from + i * 1000 / 60);
}

test('sustained slow frames lower resolution within its bounds', () => {
  const budget = new RenderBudget();
  sample(budget, 35, 0);
  assert.equal(budget.scale, 0.9);
  sample(budget, 35, 1000, 600);
  assert.equal(budget.scale, 0.5);
});

test('headroom recovers resolution gradually without exceeding the chosen scale', () => {
  const budget = new RenderBudget();
  sample(budget, 40, 0, 600);
  const low = budget.scale;
  sample(budget, 6, 10000, 60);
  assert.equal(budget.scale, low + 0.05);
  sample(budget, 6, 11000, 1200);
  assert.equal(budget.scale, 1);
});

test('normal frames, invalid samples, and a short spike do not churn render targets', () => {
  const budget = new RenderBudget();
  sample(budget, 16, 0);
  budget.observe(100, 1001);
  for (const value of [NaN, Infinity, 0, -1]) budget.observe(value, 1002);
  sample(budget, 16, 1010, 120);
  assert.equal(budget.scale, 1);
});

test('the cooldown and minimum sample count prevent repeated changes', () => {
  const budget = new RenderBudget();
  for (let i = 0; i < 11; i++) budget.observe(40, 2000);
  assert.equal(budget.scale, 1);
  budget.observe(40, 2000);
  assert.equal(budget.scale, 0.9);
  sample(budget, 40, 2000, 30);
  assert.equal(budget.scale, 0.9);
  budget.reset(2500);
  assert.equal(budget.scale, 1);
  assert.equal(budget.samples, 0);
});

test('each probe is visited exactly once per lighting update cycle', () => {
  for (const stride of [2, 4, 8]) {
    const seen = new Set();
    for (let frame = 0; frame < stride; frame++) {
      const { count, phase, hysteresis } = probeSchedule(frame, stride);
      assert.ok(Math.abs(hysteresis ** (2 / stride) - 0.95) < 1e-12);
      for (let i = 0; i < count; i++) {
        const slot = i * stride + phase;
        assert.ok(slot < 9216 && !seen.has(slot));
        seen.add(slot);
      }
    }
    assert.equal(seen.size, 9216);
  }
  assert.throws(() => probeSchedule(0, 3), RangeError);
});
