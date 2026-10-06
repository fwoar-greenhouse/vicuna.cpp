(function () {
    function parseCSV(text) {
        if (text === "") return [];
        const rows = [];
        let row = [], i = 0;
        const n = text.length;
        while (true) {
            let field = "";
            if (text[i] === '"') {
                i++;
                while (true) {
                    if (i >= n) throw new SyntaxError("unterminated quote");
                    if (text[i] === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; } else { i++; break; } }
                    else field += text[i++];
                }
                if (i < n && text[i] !== "," && text[i] !== "\n" && !(text[i] === "\r" && text[i + 1] === "\n")) throw new SyntaxError("text after quote");
            } else {
                while (i < n && text[i] !== "," && text[i] !== "\n" && !(text[i] === "\r" && text[i + 1] === "\n")) field += text[i++];
            }
            row.push(field);
            if (i >= n) { rows.push(row); break; }
            if (text[i] === ",") { i++; continue; }
            i += text[i] === "\r" ? 2 : 1;
            rows.push(row); row = [];
            if (i >= n) break;
        }
        return rows;
    }
    return { parseCSV };
})()
