(function (mod, t) {
    t.test("parse simple", () => { t.equal(mod.parseDuration("15s"), 15); t.equal(mod.parseDuration("1h"), 3600); t.equal(mod.parseDuration("2d"), 172800); });
    t.test("parse combined", () => { t.equal(mod.parseDuration("1d2h30m15s"), 95415); t.equal(mod.parseDuration("1h5s"), 3605); t.equal(mod.parseDuration("90m"), 5400); t.equal(mod.parseDuration("0s"), 0); });
    t.test("parse errors", () => { for (const s of ["", "1x", "5", "h", "1m1h", "1h1h", "1 h", "-1s", "1.5h", "1hm", " 1s"]) t.throws(() => mod.parseDuration(s), `parseDuration(${JSON.stringify(s)})`); });
    t.test("format", () => { t.equal(mod.formatDuration(0), "0s"); t.equal(mod.formatDuration(5400), "1h30m"); t.equal(mod.formatDuration(86401), "1d1s"); t.equal(mod.formatDuration(95415), "1d2h30m15s"); t.equal(mod.formatDuration(59), "59s"); });
    t.test("format errors", () => { for (const n of [-1, 1.5, NaN]) { let e; try { mod.formatDuration(n); } catch (x) { e = x; } t.ok(e instanceof RangeError, `formatDuration(${n})`); } });
    t.test("round trip", () => { for (let n = 0; n < 200000; n += 997) t.equal(mod.parseDuration(mod.formatDuration(n)), n, `n=${n}`); });
})
