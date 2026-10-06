(function () {
    const U = [["d", 86400], ["h", 3600], ["m", 60], ["s", 1]];
    function parseDuration(s) {
        if (typeof s !== "string" || !/^(\d+d)?(\d+h)?(\d+m)?(\d+s)?$/.test(s) || s === "") throw new Error("bad duration");
        let total = 0;
        for (const m of s.matchAll(/(\d+)([dhms])/g)) total += Number(m[1]) * U.find((u) => u[0] === m[2])[1];
        return total;
    }
    function formatDuration(n) {
        if (!Number.isInteger(n) || n < 0) throw new RangeError("bad seconds");
        if (n === 0) return "0s";
        let out = "";
        for (const [u, v] of U) { const q = Math.floor(n / v); if (q) { out += q + u; n -= q * v; } }
        return out;
    }
    return { parseDuration, formatDuration };
})()
