(function () {
    function createLRU(capacity) {
        if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("capacity");
        const m = new Map();
        return {
            get(k) { if (!m.has(k)) return undefined; const v = m.get(k); m.delete(k); m.set(k, v); return v; },
            set(k, v) { m.delete(k); m.set(k, v); if (m.size > capacity) m.delete(m.keys().next().value); },
            has(k) { return m.has(k); },
            delete(k) { return m.delete(k); },
            size() { return m.size; },
            keys() { return [...m.keys()].reverse(); },
        };
    }
    return { createLRU };
})()
