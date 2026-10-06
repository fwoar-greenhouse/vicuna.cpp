(function (mod, t) {
    const near = (a, b, m) => t.ok(Math.abs(a - b) < 1e-9, m ?? `expected ${b}, got ${a}`);
    t.test("numbers", () => { t.equal(mod.evaluate("42"), 42); near(mod.evaluate(".5"), 0.5); near(mod.evaluate("10.25"), 10.25); });
    t.test("precedence", () => { t.equal(mod.evaluate("2+3*4"), 14); t.equal(mod.evaluate("(2+3)*4"), 20); t.equal(mod.evaluate("10-4-3"), 3); t.equal(mod.evaluate("2*3/4*8"), 12); });
    t.test("power right assoc", () => { t.equal(mod.evaluate("2^3^2"), 512); t.equal(mod.evaluate("2^-1"), 0.5); });
    t.test("unary", () => { t.equal(mod.evaluate("-2^2"), -4); t.equal(mod.evaluate("(-2)^2"), 4); t.equal(mod.evaluate("2*-3"), -6); t.equal(mod.evaluate("--3"), 3); t.equal(mod.evaluate("+4"), 4); t.equal(mod.evaluate("1 - -1"), 2); });
    t.test("whitespace", () => t.equal(mod.evaluate("  ( 1 +\t2 ) *\n3 "), 9));
    t.test("syntax errors", () => {
        for (const s of ["", "   ", "1+", "(1+2", "1+2)", "2 3", "1 + * 2", "abc", "1..2", "()", "4^"]) {
            let e; try { mod.evaluate(s); } catch (x) { e = x; }
            t.ok(e instanceof SyntaxError, `expected SyntaxError for ${JSON.stringify(s)}, got ${e}`);
        }
    });
    t.test("division by zero", () => { let e; try { mod.evaluate("1/(2-2)"); } catch (x) { e = x; } t.ok(e instanceof RangeError, `got ${e}`); });
    t.test("long expression", () => { const s = Array.from({ length: 1000 }, (_, i) => String(i)).join("+"); t.equal(mod.evaluate(s), 499500); });
})
