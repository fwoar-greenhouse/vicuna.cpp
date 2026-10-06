(function (mod, t) {
    t.test("empty", () => t.deepEqual(mod.mergeIntervals([]), []));
    t.test("overlap", () => t.deepEqual(mod.mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]]));
    t.test("touching merges", () => t.deepEqual(mod.mergeIntervals([[1, 2], [2, 5]]), [[1, 5]]));
    t.test("adjacent integers do not merge", () => t.deepEqual(mod.mergeIntervals([[1, 2], [3, 4]]), [[1, 2], [3, 4]]));
    t.test("unsorted and nested", () => t.deepEqual(mod.mergeIntervals([[5, 7], [1, 10], [2, 3]]), [[1, 10]]));
    t.test("negative numbers", () => t.deepEqual(mod.mergeIntervals([[-5, -1], [-2, 0], [3, 3]]), [[-5, 0], [3, 3]]));
    t.test("input not modified", () => { const inp = [[3, 4], [1, 2]]; const copy = JSON.stringify(inp); mod.mergeIntervals(inp); t.equal(JSON.stringify(inp), copy); });
    t.test("result does not alias input", () => { const inp = [[1, 2]]; const r = mod.mergeIntervals(inp); r[0][1] = 99; t.equal(inp[0][1], 2); });
    t.test("bad interval", () => { let e; try { mod.mergeIntervals([[3, 1]]); } catch (x) { e = x; } t.ok(e instanceof TypeError, "TypeError"); });
    t.test("insert middle", () => t.deepEqual(mod.insertInterval([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]), [[1, 2], [3, 10], [12, 16]]));
    t.test("insert ends", () => { t.deepEqual(mod.insertInterval([[3, 5]], [0, 1]), [[0, 1], [3, 5]]); t.deepEqual(mod.insertInterval([[3, 5]], [7, 9]), [[3, 5], [7, 9]]); });
    t.test("insert into empty", () => t.deepEqual(mod.insertInterval([], [2, 4]), [[2, 4]]));
    t.test("insert does not modify input", () => { const inp = [[1, 5]]; mod.insertInterval(inp, [2, 9]); t.deepEqual(inp, [[1, 5]]); });
})
