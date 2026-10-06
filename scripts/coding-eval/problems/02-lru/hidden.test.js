(function (mod, t) {
    t.test("capacity validation", () => { for (const c of [0, -1, 1.5, NaN]) t.throws(() => mod.createLRU(c), `capacity ${c}`); });
    t.test("basic get/set", () => { const c = mod.createLRU(2); c.set("a", 1); c.set("b", 2); t.equal(c.get("a"), 1); t.equal(c.get("x"), undefined); t.equal(c.size(), 2); });
    t.test("evicts least recently used", () => {
        const c = mod.createLRU(2); c.set("a", 1); c.set("b", 2); c.get("a"); c.set("c", 3);
        t.equal(c.has("b"), false); t.equal(c.has("a"), true); t.equal(c.has("c"), true);
    });
    t.test("update moves to front", () => { const c = mod.createLRU(2); c.set("a", 1); c.set("b", 2); c.set("a", 10); c.set("c", 3); t.equal(c.get("a"), 10); t.equal(c.has("b"), false); });
    t.test("has does not change order", () => { const c = mod.createLRU(2); c.set("a", 1); c.set("b", 2); c.has("a"); c.set("c", 3); t.equal(c.has("a"), false); });
    t.test("keys order", () => { const c = mod.createLRU(3); c.set(1, "x"); c.set(2, "y"); c.set(3, "z"); c.get(1); t.deepEqual(c.keys(), [1, 3, 2]); });
    t.test("delete", () => { const c = mod.createLRU(2); c.set("a", 1); t.equal(c.delete("a"), true); t.equal(c.delete("a"), false); t.equal(c.size(), 0); });
    t.test("object and NaN keys", () => { const c = mod.createLRU(3); const k = {}; c.set(k, 1); c.set(NaN, 2); t.equal(c.get(k), 1); t.equal(c.get(NaN), 2); t.equal(c.get({}), undefined); });
    t.test("capacity 1", () => { const c = mod.createLRU(1); c.set("a", 1); c.set("b", 2); t.deepEqual(c.keys(), ["b"]); });
    t.test("many operations stay fast", () => { const c = mod.createLRU(1000); for (let i = 0; i < 200000; i++) { c.set(i, i); c.get(i - 500); } t.equal(c.size(), 1000); });
})
