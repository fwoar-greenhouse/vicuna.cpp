(function () {
    function evaluate(src) {
        const toks = [];
        for (let i = 0; i < src.length;) {
            const c = src[i];
            if (/\s/.test(c)) { i++; continue; }
            if (/[0-9.]/.test(c)) {
                let j = i; while (j < src.length && /[0-9.]/.test(src[j])) j++;
                const s = src.slice(i, j);
                if (!/^(\d+\.?\d*|\.\d+)$/.test(s)) throw new SyntaxError("bad number");
                toks.push({ t: "n", v: parseFloat(s) }); i = j; continue;
            }
            if ("+-*/^()".includes(c)) { toks.push({ t: c }); i++; continue; }
            throw new SyntaxError("bad char " + c);
        }
        let p = 0;
        const peek = () => toks[p], eat = (t) => { if (!toks[p] || toks[p].t !== t) throw new SyntaxError("expected " + t); p++; };
        function expr() { let v = term(); while (peek() && (peek().t === "+" || peek().t === "-")) { const o = toks[p++].t; const r = term(); v = o === "+" ? v + r : v - r; } return v; }
        function term() { let v = unary(); while (peek() && (peek().t === "*" || peek().t === "/")) { const o = toks[p++].t; const r = unary(); if (o === "/") { if (r === 0) throw new RangeError("div by zero"); v = v / r; } else v = v * r; } return v; }
        function unary() { if (peek() && (peek().t === "-" || peek().t === "+")) { const o = toks[p++].t; const v = unary(); return o === "-" ? -v : v; } return power(); }
        function power() { const b = atom(); if (peek() && peek().t === "^") { p++; const e = unary(); return Math.pow(b, e); } return b; }
        function atom() { const k = peek(); if (!k) throw new SyntaxError("unexpected end"); if (k.t === "n") { p++; return k.v; } if (k.t === "(") { p++; const v = expr(); eat(")"); return v; } throw new SyntaxError("unexpected " + k.t); }
        const v = expr();
        if (p !== toks.length) throw new SyntaxError("trailing input");
        return v;
    }
    return { evaluate };
})()
