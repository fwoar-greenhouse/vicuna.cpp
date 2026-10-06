(function () {
    function parseMoney(s) {
        if (typeof s !== "string") throw new SyntaxError("not a string");
        let neg = false, body = s;
        const paren = body.match(/^\((.*)\)$/);
        if (paren) { neg = true; body = paren[1]; }
        else if (body.startsWith("-")) { neg = true; body = body.slice(1); }
        if (body.startsWith("$")) body = body.slice(1);
        const m = body.match(/^(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?$/);
        if (!m) throw new SyntaxError("bad amount");
        const cents = Number(m[1].replace(/,/g, "")) * 100 + (m[2] ? Number(m[2]) : 0);
        return neg ? -cents : cents;
    }
    function formatMoney(c) {
        if (!Number.isInteger(c)) throw new RangeError("not integer");
        const a = Math.abs(c);
        const dollars = String(Math.floor(a / 100)).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
        return (c < 0 ? "-" : "") + "$" + dollars + "." + String(a % 100).padStart(2, "0");
    }
    function allocate(cents, ratios) {
        if (!Number.isInteger(cents) || !Array.isArray(ratios) || !ratios.length || ratios.some((r) => !(r >= 0)) || !ratios.some((r) => r > 0)) throw new RangeError("bad input");
        const total = ratios.reduce((a, b) => a + b, 0), a = Math.abs(cents);
        const shares = ratios.map((r) => a * r / total);
        const parts = shares.map(Math.floor);
        let left = a - parts.reduce((x, y) => x + y, 0);
        const order = shares.map((s, i) => [s - Math.floor(s), i]).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
        for (let k = 0; k < left; k++) parts[order[k][1]]++;
        return cents < 0 ? parts.map((p) => -p || 0) : parts;
    }
    return { parseMoney, formatMoney, allocate };
})()
