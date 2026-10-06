(function (mod, t) {
    function checkCycle(graph, label) {
        let e; try { mod.topoSort(graph); } catch (x) { e = x; }
        t.ok(e instanceof Error, `${label}: expected an Error`);
        const c = e.cycle;
        t.ok(Array.isArray(c) && c.length >= 2 && c[0] === c[c.length - 1], `${label}: bad cycle ${JSON.stringify(c)}`);
        for (let i = 0; i + 1 < c.length; i++) t.ok((graph[c[i]] || []).includes(c[i + 1]), `${label}: ${c[i]} does not depend on ${c[i + 1]}`);
        t.equal(new Set(c.slice(0, -1)).size, c.length - 1, `${label}: cycle repeats a node`);
    }
    t.test("simple chain", () => t.deepEqual(mod.topoSort({ c: ["b"], b: ["a"] }), ["a", "b", "c"]));
    t.test("ties by name", () => t.deepEqual(mod.topoSort({ d: ["b", "c"], b: ["a"], c: ["a"], a: [] }), ["a", "b", "c", "d"]));
    t.test("independent nodes", () => t.deepEqual(mod.topoSort({ z: [], y: [], x: [] }), ["x", "y", "z"]));
    t.test("dependency-only nodes", () => t.deepEqual(mod.topoSort({ app: ["lib", "util"], lib: ["util"] }), ["util", "lib", "app"]));
    t.test("smallest ready first, not alphabetical overall", () => t.deepEqual(mod.topoSort({ a: ["z"], b: [] }), ["b", "z", "a"]));
    t.test("empty graph", () => t.deepEqual(mod.topoSort({}), []));
    t.test("cycle of three", () => checkCycle({ a: ["b"], b: ["c"], c: ["a"], d: [] }, "three"));
    t.test("self dependency", () => { checkCycle({ a: ["a"] }, "self"); });
    t.test("cycle behind a chain", () => checkCycle({ start: ["x"], x: ["y"], y: ["z"], z: ["x"] }, "behind chain"));
    t.test("large graph", () => {
        const g = {}; for (let i = 1; i < 3000; i++) g["n" + String(i).padStart(5, "0")] = ["n" + String(i - 1).padStart(5, "0")];
        const r = mod.topoSort(g); t.equal(r.length, 3000); t.equal(r[0], "n00000"); t.equal(r[2999], "n02999");
    });
})
