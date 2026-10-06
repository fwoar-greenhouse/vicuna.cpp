(function () {
    function justify(text, width) {
        if (!Number.isInteger(width) || width < 1) throw new RangeError("width");
        const words = text.split(/\s+/).filter(Boolean);
        const lines = [];
        let cur = [];
        let len = 0;
        for (const w of words) {
            if (cur.length && len + 1 + w.length > width) { lines.push(cur); cur = []; len = 0; }
            len = cur.length ? len + 1 + w.length : w.length;
            cur.push(w);
        }
        if (cur.length) lines.push(cur);
        return lines.map((ws, i) => {
            const last = i === lines.length - 1;
            const chars = ws.reduce((a, w) => a + w.length, 0);
            if (last || ws.length === 1) {
                const s = ws.join(" ");
                return s.length >= width ? s : s + " ".repeat(width - s.length);
            }
            const gaps = ws.length - 1, spaces = width - chars;
            const base = Math.floor(spaces / gaps), extra = spaces % gaps;
            return ws.map((w, j) => j < gaps ? w + " ".repeat(base + (j < extra ? 1 : 0)) : w).join("");
        });
    }
    return { justify };
})()
