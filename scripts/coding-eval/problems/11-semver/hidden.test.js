(function (mod, t) {
    t.test("basic compare", () => { t.equal(mod.compare("1.0.0", "2.0.0"), -1); t.equal(mod.compare("1.10.0", "1.9.0"), 1); t.equal(mod.compare("1.2.3", "1.2.3"), 0); });
    t.test("prerelease order", () => {
        const order = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"];
        for (let i = 0; i + 1 < order.length; i++) { t.equal(mod.compare(order[i], order[i + 1]), -1, `${order[i]} < ${order[i + 1]}`); t.equal(mod.compare(order[i + 1], order[i]), 1); }
    });
    t.test("build metadata ignored", () => t.equal(mod.compare("1.2.3+a", "1.2.3+b"), 0));
    t.test("invalid versions", () => { for (const v of ["1.2", "01.2.3", "1.2.3-", "1.2.3-01", "a.b.c", "1.2.3-a..b", ""]) { let e; try { mod.compare(v, "1.0.0"); } catch (x) { e = x; } t.ok(e instanceof TypeError, `compare(${JSON.stringify(v)})`); } });
    t.test("comparators", () => {
        t.equal(mod.satisfies("1.2.3", "1.2.3"), true); t.equal(mod.satisfies("1.2.3", "=1.2.3"), true);
        t.equal(mod.satisfies("1.2.4", ">1.2.3"), true); t.equal(mod.satisfies("1.2.3", ">1.2.3"), false);
        t.equal(mod.satisfies("1.2.3", ">=1.2.3 <1.3.0"), true); t.equal(mod.satisfies("1.3.0", ">=1.2.3 <1.3.0"), false);
    });
    t.test("caret", () => {
        t.equal(mod.satisfies("1.9.9", "^1.2.3"), true); t.equal(mod.satisfies("2.0.0", "^1.2.3"), false); t.equal(mod.satisfies("1.2.2", "^1.2.3"), false);
        t.equal(mod.satisfies("0.2.9", "^0.2.3"), true); t.equal(mod.satisfies("0.3.0", "^0.2.3"), false);
        t.equal(mod.satisfies("0.0.3", "^0.0.3"), true); t.equal(mod.satisfies("0.0.4", "^0.0.3"), false);
    });
    t.test("tilde", () => { t.equal(mod.satisfies("1.2.9", "~1.2.3"), true); t.equal(mod.satisfies("1.3.0", "~1.2.3"), false); });
    t.test("or", () => { t.equal(mod.satisfies("3.0.0", "^1.0.0 || >=3.0.0"), true); t.equal(mod.satisfies("2.5.0", "^1.0.0 || >=3.0.0"), false); });
    t.test("prerelease in ranges", () => {
        t.equal(mod.satisfies("1.3.0-beta", "^1.2.0"), false);
        t.equal(mod.satisfies("1.2.3-beta.2", ">=1.2.3-beta.1"), true);
        t.equal(mod.satisfies("1.2.4-beta.2", ">=1.2.3-beta.1"), false);
        t.equal(mod.satisfies("1.2.3-beta.2", ">=1.2.3-beta.1 <1.3.0"), true);
    });
})
