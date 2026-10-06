(function () {
    function diffLines(a, b) {
        const n = a.length, m = b.length;
        const L = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
        for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
            L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
        const out = [];
        let i = 0, j = 0, dels = [], ins = [];
        const flush = () => { out.push(...dels, ...ins); dels = []; ins = []; };
        while (i < n || j < m) {
            if (i < n && j < m && a[i] === b[j]) { flush(); out.push({ op: "=", line: a[i] }); i++; j++; }
            else if (j < m && (i === n || L[i][j + 1] >= L[i + 1][j])) { ins.push({ op: "+", line: b[j] }); j++; }
            else { dels.push({ op: "-", line: a[i] }); i++; }
        }
        flush();
        return out;
    }
    return { diffLines };
})()
