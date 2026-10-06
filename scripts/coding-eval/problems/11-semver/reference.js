(function () {
    const NUM = "(0|[1-9]\\d*)", ID = "(?:0|[1-9]\\d*|\\d*[A-Za-z-][0-9A-Za-z-]*)";
    const RE = new RegExp(`^${NUM}\\.${NUM}\\.${NUM}(?:-(${ID}(?:\\.${ID})*))?(?:\\+[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*)?$`);
    function parse(v) {
        const m = typeof v === "string" && v.match(RE);
        if (!m) throw new TypeError("invalid version " + v);
        return { nums: [+m[1], +m[2], +m[3]], pre: m[4] ? m[4].split(".") : [] };
    }
    function cmpId(a, b) {
        const an = /^\d+$/.test(a), bn = /^\d+$/.test(b);
        if (an && bn) return Math.sign(+a - +b);
        if (an) return -1;
        if (bn) return 1;
        return a < b ? -1 : a > b ? 1 : 0;
    }
    function cmpParsed(x, y) {
        for (let i = 0; i < 3; i++) if (x.nums[i] !== y.nums[i]) return Math.sign(x.nums[i] - y.nums[i]);
        if (!x.pre.length && !y.pre.length) return 0;
        if (!x.pre.length) return 1;
        if (!y.pre.length) return -1;
        for (let i = 0; i < Math.min(x.pre.length, y.pre.length); i++) { const c = cmpId(x.pre[i], y.pre[i]); if (c) return c; }
        return Math.sign(x.pre.length - y.pre.length);
    }
    function compare(a, b) { return cmpParsed(parse(a), parse(b)); }
    function expand(c) {
        const m = c.match(/^(>=|<=|>|<|=|\^|~)?(.+)$/);
        const op = m[1] || "=", v = parse(m[2]);
        const [M, mi, p] = v.nums;
        const at = (nums) => ({ nums, pre: [] });
        if (op === "^") { const up = M > 0 ? [M + 1, 0, 0] : mi > 0 ? [0, mi + 1, 0] : [0, 0, p + 1]; return [[">=", v], ["<", at(up)]]; }
        if (op === "~") return [[">=", v], ["<", at([M, mi + 1, 0])]];
        return [[op, v]];
    }
    function test(op, x, v) {
        const c = cmpParsed(x, v);
        return op === "=" ? c === 0 : op === ">" ? c > 0 : op === ">=" ? c >= 0 : op === "<" ? c < 0 : c <= 0;
    }
    function satisfies(version, range) {
        const x = parse(version);
        return range.split("||").some((set) => {
            const comps = set.trim().split(/\s+/).filter(Boolean).flatMap(expand);
            if (!comps.length) return false;
            if (!comps.every(([op, v]) => test(op, x, v))) return false;
            if (!x.pre.length) return true;
            return comps.some(([, v]) => v.pre.length && v.nums.every((n, i) => n === x.nums[i]));
        });
    }
    return { compare, satisfies };
})()
