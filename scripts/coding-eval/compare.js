// Compare coding eval results across label directories (read-only).
// usage: compare.sh <label dir> [<label dir> ...] > compare.md

async function loadLabel(dir) {
    const meta = JSON.parse(await Deno.readTextFile(`${dir}/meta.json`));
    const results = [];
    for await (const p of Deno.readDir(dir)) {
        if (!p.isDirectory) continue;
        for await (const m of Deno.readDir(`${dir}/${p.name}`)) {
            if (!m.isDirectory) continue;
            for await (const r of Deno.readDir(`${dir}/${p.name}/${m.name}`)) {
                if (!r.isDirectory) continue;
                const base = `${dir}/${p.name}/${m.name}/${r.name}`;
                let res = null, error = null;
                try {
                    res = JSON.parse(await Deno.readTextFile(`${base}/result.json`));
                } catch {
                    try {
                        error = (await Deno.readTextFile(`${base}/error.txt`)).split("\n")[0];
                    } catch { /* not started */ }
                }
                if (res || error) results.push({ problem: p.name, mode: m.name, run: r.name, res, error });
            }
        }
    }
    return { dir, meta, results };
}

function cell(rs) {
    if (!rs.length) return "-";
    const ok = rs.filter((r) => r.res?.hidden.ok).length;
    const tests = rs.map((r) => (r.res ? `${r.res.hidden.passed}/${r.res.hidden.total}` : "err")).join(" ");
    return `${ok}/${rs.length} (${tests})`;
}

function stat(rs, f) {
    const v = rs.map((r) => r.res && f(r.res)).filter((x) => typeof x === "number" && isFinite(x));
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

const fmt = (x, d = 0) => (x === null ? "-" : x.toFixed(d));

async function main() {
    if (!Deno.args.length) {
        console.error("usage: compare.sh <label dir> [<label dir> ...]");
        return 2;
    }
    const labels = [];
    for (const d of Deno.args) labels.push(await loadLabel(d.replace(/\/+$/, "")));
    const names = labels.map((l) => l.dir.split("/").pop());
    const out = ["# Coding eval comparison", ""];
    for (const [i, l] of labels.entries()) {
        const m = l.meta;
        out.push(`- **${names[i]}**: ${m.model} @ ${m.endpoint} (temperature ${m.temperature ?? "default"}, top-p ${m.topP ?? "default"}, top-k ${m.topK ?? "default"}, reasoning ${m.reasoningEffort ?? "default"}, ${m.runs} runs)`);
    }
    out.push("");
    const problems = [...new Set(labels.flatMap((l) => l.results.map((r) => r.problem)))].sort();
    for (const mode of ["plain", "agentic"]) {
        if (!labels.some((l) => l.results.some((r) => r.mode === mode))) continue;
        out.push(`## ${mode}`, "", `Cells: attempts passing all hidden tests / attempts (hidden tests passed per attempt).`, "");
        out.push(`| problem | ${names.join(" | ")} |`, `|---|${names.map(() => "---").join("|")}|`);
        for (const p of problems) out.push(`| ${p} | ${labels.map((l) => cell(l.results.filter((r) => r.problem === p && r.mode === mode))).join(" | ")} |`);
        out.push(`| **total** | ${labels.map((l) => cell(l.results.filter((r) => r.mode === mode)).replace(/ \(.*\)$/, "")).join(" | ")} |`, "");
        const rows = [
            ["requests per attempt", (r) => r.stats?.requests, 1],
            ["completion tokens per attempt", (r) => r.stats?.completionTokens, 0],
            ["wall seconds per attempt", (r) => (r.stats?.wallMs ?? NaN) / 1000, 0],
            ["gen t/s (llama-server timings)", (r) => (r.stats?.predictedMs ? r.stats.predictedN / (r.stats.predictedMs / 1000) : NaN), 1],
        ];
        if (mode === "agentic") {
            rows.push(["own tests pass on the hidden reference (share)", (r) => (r.testsVsReference ? (r.testsVsReference.ok ? 1 : 0) : NaN), 2]);
            rows.push(["own tests fail on the placeholder (share)", (r) => (r.red ? (r.red.failingAgainstPlaceholder ? 1 : 0) : NaN), 2]);
            rows.push(["green: own tests pass (share)", (r) => (r.green ? (r.green.ownTestsPass ? 1 : 0) : NaN), 2]);
            rows.push(["refactor reverted (share)", (r) => (r.refactorReverted ? 1 : 0), 2]);
            rows.push(["test runs per attempt", (r) => r.testRuns, 1]);
        }
        out.push(`| metric | ${names.join(" | ")} |`, `|---|${names.map(() => "---:").join("|")}|`);
        for (const [label, f, d] of rows) out.push(`| ${label} | ${labels.map((l) => fmt(stat(l.results.filter((r) => r.mode === mode), f), d)).join(" | ")} |`);
        out.push("");
    }
    const errors = labels.flatMap((l, i) => l.results.filter((r) => r.error).map((r) => `- ${names[i]} ${r.problem} ${r.mode} ${r.run}: ${r.error}`));
    if (errors.length) out.push("## Errors", "", ...errors, "");
    console.log(out.join("\n"));
    return 0;
}

Deno.exit(await main());
