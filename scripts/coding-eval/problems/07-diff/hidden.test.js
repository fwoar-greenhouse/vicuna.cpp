(function (mod, t) {
    function lcs(a, b) {
        let prev = new Array(b.length + 1).fill(0);
        for (let i = 1; i <= a.length; i++) {
            const cur = [0];
            for (let j = 1; j <= b.length; j++) cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
            prev = cur;
        }
        return prev[b.length];
    }
    function check(a, b, label) {
        const ops = mod.diffLines(a, b);
        t.ok(Array.isArray(ops), `${label}: result is an array`);
        for (const o of ops) t.ok(o && ["=", "-", "+"].includes(o.op) && typeof o.line === "string", `${label}: bad op ${JSON.stringify(o)}`);
        t.deepEqual(ops.filter((o) => o.op !== "+").map((o) => o.line), a, `${label}: does not rebuild a`);
        t.deepEqual(ops.filter((o) => o.op !== "-").map((o) => o.line), b, `${label}: does not rebuild b`);
        t.equal(ops.filter((o) => o.op === "=").length, lcs(a, b), `${label}: not minimal`);
        let seenPlus = false;
        for (const o of ops) {
            if (o.op === "=") seenPlus = false;
            else if (o.op === "+") seenPlus = true;
            else t.ok(!seenPlus, `${label}: '-' after '+' in a change run`);
        }
    }
    t.test("identical", () => check(["a", "b"], ["a", "b"], "identical"));
    t.test("empty sides", () => { check([], [], "both empty"); check(["a"], [], "to empty"); check([], ["a", "b"], "from empty"); });
    t.test("replace", () => check(["a", "b", "c"], ["a", "x", "c"], "replace"));
    t.test("classic", () => check("ABCABBA".split(""), "CBABAC".split(""), "classic"));
    t.test("moves", () => check(["1", "2", "3", "4", "5"], ["5", "1", "2", "4", "3"], "moves"));
    t.test("duplicates", () => check(["a", "a", "b", "a"], ["a", "b", "a", "a"], "duplicates"));
    t.test("random", () => {
        let s = 12345; const rnd = () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
        for (let k = 0; k < 30; k++) {
            const a = Array.from({ length: Math.floor(rnd() * 15) }, () => "abcd"[Math.floor(rnd() * 4)]);
            const b = Array.from({ length: Math.floor(rnd() * 15) }, () => "abcd"[Math.floor(rnd() * 4)]);
            check(a, b, `random ${k}`);
        }
    });
    t.test("large input", () => {
        const a = Array.from({ length: 2000 }, (_, i) => "line " + (i % 50));
        const b = a.map((x, i) => (i % 7 === 0 ? "changed " + i : x));
        check(a, b, "large");
    });
})
