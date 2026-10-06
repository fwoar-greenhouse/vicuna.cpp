(function () {
    function topoSort(graph) {
        const deps = new Map();
        const add = (n) => { if (!deps.has(n)) deps.set(n, new Set()); };
        for (const [n, ds] of Object.entries(graph)) { add(n); for (const d of ds) { add(d); deps.get(n).add(d); } }
        const indeg = new Map(), users = new Map();
        for (const n of deps.keys()) { indeg.set(n, deps.get(n).size); users.set(n, []); }
        for (const [n, ds] of deps) for (const d of ds) users.get(d).push(n);
        const ready = [...deps.keys()].filter((n) => indeg.get(n) === 0).sort();
        const out = [];
        while (ready.length) {
            const n = ready.shift(); out.push(n);
            for (const u of users.get(n)) { indeg.set(u, indeg.get(u) - 1); if (indeg.get(u) === 0) { ready.push(u); ready.sort(); } }
        }
        if (out.length === deps.size) return out;
        // Walk dependency edges among the remaining nodes until a node repeats.
        const left = new Set([...deps.keys()].filter((n) => !out.includes(n)));
        let cur = [...left].sort()[0];
        const path = [], pos = new Map();
        while (!pos.has(cur)) { pos.set(cur, path.length); path.push(cur); cur = [...deps.get(cur)].filter((d) => left.has(d)).sort()[0]; }
        const err = new Error("cycle");
        err.cycle = [...path.slice(pos.get(cur)), cur];
        throw err;
    }
    return { topoSort };
})()
