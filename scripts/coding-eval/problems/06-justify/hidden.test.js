(function (mod, t) {
    t.test("classic example", () => t.deepEqual(mod.justify("This is an example of text justification.", 16),
        ["This    is    an", "example  of text", "justification.  "]));
    t.test("uneven gaps go left", () => t.deepEqual(mod.justify("What must be acknowledgment shall be", 16),
        ["What   must   be", "acknowledgment  ", "shall be        "]));
    t.test("whitespace runs", () => t.deepEqual(mod.justify("  a\tb \n c  ", 5), ["a b c"]));
    t.test("empty", () => { t.deepEqual(mod.justify("", 10), []); t.deepEqual(mod.justify("   \n ", 10), []); });
    t.test("long word", () => t.deepEqual(mod.justify("hi supercalifragilistic yo", 6), ["hi    ", "supercalifragilistic", "yo    "]));
    t.test("exact fit", () => t.deepEqual(mod.justify("ab cd ef", 5), ["ab cd", "ef   "]));
    t.test("lines have width", () => {
        const txt = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua";
        for (const w of [12, 17, 23, 40]) for (const line of mod.justify(txt, w)) t.equal(line.length, w, `width ${w}: ${JSON.stringify(line)}`);
    });
    t.test("bad width", () => { for (const w of [0, -3, 2.5]) { let e; try { mod.justify("a", w); } catch (x) { e = x; } t.ok(e instanceof RangeError, `width ${w}`); } });
})
