(function (mod, t) {
    const pairs = [[1, "I"], [4, "IV"], [9, "IX"], [14, "XIV"], [40, "XL"], [90, "XC"], [400, "CD"],
        [944, "CMXLIV"], [1994, "MCMXCIV"], [2024, "MMXXIV"], [3999, "MMMCMXCIX"]];
    t.test("toRoman known values", () => { for (const [n, s] of pairs) t.equal(mod.toRoman(n), s, `toRoman(${n})`); });
    t.test("fromRoman known values", () => { for (const [n, s] of pairs) t.equal(mod.fromRoman(s), n, `fromRoman(${s})`); });
    t.test("fromRoman lower case", () => { t.equal(mod.fromRoman("xiv"), 14); t.equal(mod.fromRoman("mcmxciv"), 1994); });
    t.test("round trip 1..3999", () => { for (let n = 1; n <= 3999; n++) t.equal(mod.fromRoman(mod.toRoman(n)), n, `n=${n}`); });
    t.test("toRoman range errors", () => { for (const n of [0, -1, 4000, 2.5, NaN]) t.throws(() => mod.toRoman(n), `toRoman(${n})`); });
    t.test("toRoman throws RangeError", () => { let e; try { mod.toRoman(0); } catch (x) { e = x; } t.ok(e instanceof RangeError, "expected RangeError"); });
    t.test("fromRoman rejects invalid", () => { for (const s of ["", "IIII", "VV", "IC", "XM", "IL", "ABC", "MMMM", "VX", "XIIII"]) t.throws(() => mod.fromRoman(s), `fromRoman(${JSON.stringify(s)})`); });
    t.test("calc operations", () => {
        t.equal(mod.calc("XIV", "*", "VII"), "XCVIII");
        t.equal(mod.calc("X", "+", "V"), "XV");
        t.equal(mod.calc("M", "-", "I"), "CMXCIX");
        t.equal(mod.calc("C", "/", "XX"), "V");
        t.equal(mod.calc("mm", "+", "cm"), "MMCM");
    });
    t.test("calc errors", () => {
        t.throws(() => mod.calc("X", "/", "III"), "inexact division");
        t.throws(() => mod.calc("V", "-", "V"), "zero result");
        t.throws(() => mod.calc("MM", "*", "II"), "too large");
        t.throws(() => mod.calc("X", "%", "II"), "bad operator");
        t.throws(() => mod.calc("IIII", "+", "I"), "bad operand");
    });
})
