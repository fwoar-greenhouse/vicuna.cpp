(function () {
    function createBucket(opts) {
        const { capacity, refillPerSecond, now } = opts ?? {};
        if (!(capacity > 0) || !(refillPerSecond >= 0) || typeof now !== "function") throw new RangeError("bad options");
        let tokens = capacity, last = now();
        const refill = () => { const t = now(); tokens = Math.min(capacity, tokens + (t - last) / 1000 * refillPerSecond); last = t; };
        const check = (n) => { if (!(n > 0) || n > capacity) throw new RangeError("bad n"); };
        return {
            tryTake(n = 1) { check(n); refill(); if (tokens + 1e-9 >= n) { tokens = Math.max(0, tokens - n); return true; } return false; },
            available() { refill(); return tokens; },
            msUntil(n = 1) { check(n); refill(); if (tokens >= n) return 0; if (refillPerSecond === 0) return Infinity; return (n - tokens) / refillPerSecond * 1000; },
        };
    }
    return { createBucket };
})()
