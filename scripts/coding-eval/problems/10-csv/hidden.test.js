(function (mod, t) {
    t.test("empty", () => t.deepEqual(mod.parseCSV(""), []));
    t.test("simple", () => t.deepEqual(mod.parseCSV("a,b,c\n1,2,3"), [["a", "b", "c"], ["1", "2", "3"]]));
    t.test("trailing newline", () => { t.deepEqual(mod.parseCSV("a,b\n"), [["a", "b"]]); t.deepEqual(mod.parseCSV("a,b\r\n"), [["a", "b"]]); });
    t.test("crlf", () => t.deepEqual(mod.parseCSV("a,b\r\nc,d"), [["a", "b"], ["c", "d"]]));
    t.test("empty fields", () => t.deepEqual(mod.parseCSV(",a,,\n,"), [["", "a", "", ""], ["", ""]]));
    t.test("empty line in the middle", () => t.deepEqual(mod.parseCSV("a\n\nb"), [["a"], [""], ["b"]]));
    t.test("quoted", () => t.deepEqual(mod.parseCSV('"a,b","c\nd","e""f"'), [["a,b", "c\nd", 'e"f']]));
    t.test("quoted empty", () => t.deepEqual(mod.parseCSV('"",x'), [["", "x"]]));
    t.test("quote inside unquoted field", () => t.deepEqual(mod.parseCSV('ab"c,d'), [['ab"c', "d"]]));
    t.test("spaces kept", () => t.deepEqual(mod.parseCSV(" a , b "), [[" a ", " b "]]));
    t.test("lone carriage return is data", () => t.deepEqual(mod.parseCSV("a\rb,c"), [["a\rb", "c"]]));
    t.test("errors", () => {
        for (const s of ['"abc', 'a,"b', '"a"b', '"a" ,c']) { let e; try { mod.parseCSV(s); } catch (x) { e = x; } t.ok(e instanceof SyntaxError, `parseCSV(${JSON.stringify(s)})`); }
    });
    t.test("large input", () => { const line = "x,\"y,z\",w\n"; const r = mod.parseCSV(line.repeat(20000)); t.equal(r.length, 20000); t.deepEqual(r[19999], ["x", "y,z", "w"]); });
})
