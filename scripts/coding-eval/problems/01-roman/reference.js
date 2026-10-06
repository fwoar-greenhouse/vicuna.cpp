(function () {
    const TABLE = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"],
        [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
    function toRoman(n) {
        if (!Number.isInteger(n) || n < 1 || n > 3999) throw new RangeError("out of range");
        let out = "";
        for (const [v, s] of TABLE) while (n >= v) { out += s; n -= v; }
        return out;
    }
    function fromRoman(str) {
        if (typeof str !== "string" || str.length === 0) throw new Error("empty");
        const s = str.toUpperCase();
        const V = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
        let total = 0;
        for (let i = 0; i < s.length; i++) {
            const v = V[s[i]];
            if (v === undefined) throw new Error("bad char");
            const next = V[s[i + 1]];
            total += next !== undefined && next > v ? -v : v;
        }
        if (total < 1 || total > 3999 || toRoman(total) !== s) throw new Error("not canonical");
        return total;
    }
    function calc(a, op, b) {
        const x = fromRoman(a), y = fromRoman(b);
        let r;
        if (op === "+") r = x + y;
        else if (op === "-") r = x - y;
        else if (op === "*") r = x * y;
        else if (op === "/") { if (x % y !== 0) throw new Error("not exact"); r = x / y; }
        else throw new Error("bad op");
        if (r < 1 || r > 3999) throw new Error("out of range");
        return toRoman(r);
    }
    return { toRoman, fromRoman, calc };
})()
