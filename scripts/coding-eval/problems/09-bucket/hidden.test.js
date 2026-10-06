(function (mod, t) {
    const clock = () => { let ms = 1000; return { now: () => ms, advance: (d) => { ms += d; } }; };
    const near = (a, b, m) => t.ok(Math.abs(a - b) < 1e-6, (m ?? "") + ` expected ${b}, got ${a}`);
    t.test("starts full", () => { const c = clock(); const b = mod.createBucket({ capacity: 5, refillPerSecond: 1, now: c.now }); near(b.available(), 5); });
    t.test("take until empty", () => {
        const c = clock(); const b = mod.createBucket({ capacity: 3, refillPerSecond: 1, now: c.now });
        t.equal(b.tryTake(), true); t.equal(b.tryTake(2), true); t.equal(b.tryTake(), false); near(b.available(), 0);
    });
    t.test("failed take changes nothing", () => { const c = clock(); const b = mod.createBucket({ capacity: 3, refillPerSecond: 1, now: c.now }); b.tryTake(2); t.equal(b.tryTake(2), false); near(b.available(), 1); });
    t.test("refill is continuous", () => {
        const c = clock(); const b = mod.createBucket({ capacity: 10, refillPerSecond: 2, now: c.now });
        b.tryTake(10); c.advance(250); near(b.available(), 0.5); c.advance(250); t.equal(b.tryTake(), true); near(b.available(), 0);
    });
    t.test("caps at capacity", () => { const c = clock(); const b = mod.createBucket({ capacity: 4, refillPerSecond: 100, now: c.now }); b.tryTake(1); c.advance(60000); near(b.available(), 4); });
    t.test("msUntil", () => {
        const c = clock(); const b = mod.createBucket({ capacity: 10, refillPerSecond: 4, now: c.now });
        t.equal(b.msUntil(3), 0); b.tryTake(10); near(b.msUntil(1), 250); near(b.msUntil(2), 500);
        c.advance(100); near(b.msUntil(1), 150);
    });
    t.test("zero refill", () => { const c = clock(); const b = mod.createBucket({ capacity: 2, refillPerSecond: 0, now: c.now }); b.tryTake(2); c.advance(1e9); t.equal(b.tryTake(), false); t.equal(b.msUntil(1), Infinity); });
    t.test("option validation", () => {
        const now = () => 0;
        for (const o of [{ capacity: 0, refillPerSecond: 1, now }, { capacity: 1, refillPerSecond: -1, now }, { capacity: 1, refillPerSecond: 1 }]) {
            let e; try { mod.createBucket(o); } catch (x) { e = x; } t.ok(e instanceof RangeError, `options ${JSON.stringify(o)}`);
        }
    });
    t.test("n validation", () => {
        const c = clock(); const b = mod.createBucket({ capacity: 3, refillPerSecond: 1, now: c.now });
        for (const n of [0, -1, 4]) { let e; try { b.tryTake(n); } catch (x) { e = x; } t.ok(e instanceof RangeError, `tryTake(${n})`); }
    });
    t.test("uses only the injected clock", () => {
        let calls = 0; const b = mod.createBucket({ capacity: 1, refillPerSecond: 1, now: () => { calls++; return 5000; } });
        b.available(); t.ok(calls >= 1, "now() was not called");
    });
})
