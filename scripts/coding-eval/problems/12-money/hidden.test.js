(function (mod, t) {
    t.test("parse", () => {
        const cases = [["$1,234.56", 123456], ["-$0.05", -5], ["12", 1200], ["(3.10)", -310], ["($1,234.50)", -123450], ["0.99", 99], ["$1,000,000", 100000000], ["-12.00", -1200]];
        for (const [s, v] of cases) t.equal(mod.parseMoney(s), v, `parseMoney(${s})`);
    });
    t.test("parse errors", () => {
        for (const s of ["1.5", "1,23", "$-1", "abc", "", "1.234", "12,", ",123", "$", "1,2345", "((1))", "--1"]) { let e; try { mod.parseMoney(s); } catch (x) { e = x; } t.ok(e instanceof SyntaxError, `parseMoney(${JSON.stringify(s)})`); }
    });
    t.test("format", () => { t.equal(mod.formatMoney(123456), "$1,234.56"); t.equal(mod.formatMoney(-5), "-$0.05"); t.equal(mod.formatMoney(0), "$0.00"); t.equal(mod.formatMoney(100000000), "$1,000,000.00"); });
    t.test("format errors", () => { let e; try { mod.formatMoney(1.5); } catch (x) { e = x; } t.ok(e instanceof RangeError); });
    t.test("allocate examples", () => {
        t.deepEqual(mod.allocate(100, [1, 1, 1]), [34, 33, 33]);
        t.deepEqual(mod.allocate(5, [1, 1]), [3, 2]);
        t.deepEqual(mod.allocate(1000, [70, 30]), [700, 300]);
        t.deepEqual(mod.allocate(-100, [1, 1, 1]), [-34, -33, -33]);
        t.deepEqual(mod.allocate(10, [0, 1]), [0, 10]);
        t.deepEqual(mod.allocate(7, [3, 3, 1]), [3, 3, 1]);
        t.deepEqual(mod.allocate(11, [3, 3, 4]), [3, 3, 5]);
    });
    t.test("allocate sums exactly", () => {
        let s = 7; const rnd = () => (s = (s * 48271) % 2147483647) / 2147483647;
        for (let k = 0; k < 200; k++) {
            const cents = Math.floor(rnd() * 100000) - 50000, ratios = Array.from({ length: 1 + Math.floor(rnd() * 6) }, () => Math.floor(rnd() * 10));
            if (!ratios.some((r) => r > 0)) ratios[0] = 1;
            const parts = mod.allocate(cents, ratios);
            t.equal(parts.reduce((a, b) => a + b, 0), cents, `sum for ${cents} ${JSON.stringify(ratios)}`);
            t.ok(parts.every(Number.isInteger), "integer parts");
        }
    });
    t.test("allocate errors", () => { for (const r of [[], [0, 0], [-1, 2], [NaN]]) { let e; try { mod.allocate(100, r); } catch (x) { e = x; } t.ok(e instanceof RangeError, `ratios ${JSON.stringify(r)}`); } });
})
