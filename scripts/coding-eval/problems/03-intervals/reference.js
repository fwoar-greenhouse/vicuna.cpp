(function () {
    function mergeIntervals(list) {
        for (const [a, b] of list) if (a > b) throw new TypeError("bad interval");
        const s = list.map(([a, b]) => [a, b]).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
        const out = [];
        for (const iv of s) {
            const last = out[out.length - 1];
            if (last && iv[0] <= last[1]) last[1] = Math.max(last[1], iv[1]);
            else out.push(iv);
        }
        return out;
    }
    function insertInterval(list, iv) { return mergeIntervals([...list, iv]); }
    return { mergeIntervals, insertInterval };
})()
