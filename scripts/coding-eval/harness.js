// Coding eval harness: asks a model (any OpenAI-compatible chat endpoint) to solve the problems in
// ./problems, runs the answers in the sandbox (sandbox.js) and writes transcripts and results.
//
// Run it through run.sh, which grants only the permissions it needs.
//
// Modes:
//   plain    one request per attempt; the module is taken from the reply
//   agentic  red / green / refactor phases through tools. The tools are fixed capabilities on the current
//            attempt's test suite and module: the model never sees or names files or commands.
//
// Every attempt starts a new conversation. Hidden tests are never shown to the model.

import { checkSource } from "./checker.js";

const HERE = new URL(".", import.meta.url).pathname;

// ---------------------------------------------------------------------------------------------
// arguments

function parseArgs(argv) {
    const opts = {
        runs: 3, mode: "both", out: "results", maxTokens: 65536, maxSteps: 12, sandboxTimeoutMs: 20000,
        requestTimeoutS: 1800, parallel: 1, temperature: undefined, topP: undefined,
    };
    const flags = new Set(["self-check", "help", "resume", "regrade"]);
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (!a.startsWith("--")) throw new Error(`unexpected argument ${a}`);
        const key = a.slice(2);
        if (flags.has(key)) { opts[key] = true; continue; }
        const v = argv[++i];
        if (v === undefined) throw new Error(`missing value for ${a}`);
        switch (key) {
            case "endpoint": opts.endpoint = v; break;
            case "model": opts.model = v; break;
            case "token": opts.token = v; break;
            case "out": opts.out = v; break;
            case "label": opts.label = v; break;
            case "runs": opts.runs = parseInt(v, 10); break;
            case "mode": opts.mode = v; break;
            case "problems": opts.problems = v.split(",").map((s) => s.trim()).filter(Boolean); break;
            case "temperature": opts.temperature = parseFloat(v); break;
            case "top-p": opts.topP = parseFloat(v); break;
            case "top-k": opts.topK = parseInt(v, 10); break;
            case "reasoning-effort": opts.reasoningEffort = v; break;
            case "max-tokens": opts.maxTokens = parseInt(v, 10); break;
            case "max-steps": opts.maxSteps = parseInt(v, 10); break;
            case "parallel": opts.parallel = parseInt(v, 10); break;
            case "sandbox-timeout-ms": opts.sandboxTimeoutMs = parseInt(v, 10); break;
            case "request-timeout-s": opts.requestTimeoutS = parseInt(v, 10); break;
            case "deno": opts.deno = v; break;
            case "bwrap": opts.bwrap = v; break;
            default: throw new Error(`unknown option ${a}`);
        }
    }
    return opts;
}

const USAGE = `usage: run.sh --endpoint URL --model ID [options]
  --endpoint URL        OpenAI-compatible base URL (http://host:8001, https://openrouter.ai/api/v1, ...)
  --model ID            model id sent in requests
  --token TOKEN         API token (prefer the CODING_EVAL_TOKEN environment variable)
  --out DIR             output directory (default: results)
  --label NAME          subdirectory for this run (default: model id + time)
  --runs N              attempts per problem and mode (default: 3)
  --mode M              plain, agentic or both (default: both)
  --problems a,b        only these problem ids (default: all)
  --temperature X       sampling temperature (default: server default)
  --top-p X             top-p (default: server default)
  --top-k N             top-k (default: server default)
  --reasoning-effort E  low, medium or high (default: server default)
  --resume              keep finished attempts in the label directory and run only the missing ones
  --regrade             re-extract and re-grade the saved plain-mode replies of --label (no requests)
  --max-tokens N        max tokens per reply (default: 65536)
  --max-steps N         max model calls per agentic phase (default: 12)
  --parallel N          attempts in flight at once (default: 1)
  --self-check          check the problems against their reference solutions; no API calls`;

// ---------------------------------------------------------------------------------------------
// problems

async function loadProblems(filter) {
    const dir = `${HERE}problems`;
    const ids = [];
    for await (const e of Deno.readDir(dir)) if (e.isDirectory) ids.push(e.name);
    ids.sort();
    const out = [];
    for (const id of ids) {
        if (filter && !filter.includes(id)) continue;
        const read = (f) => Deno.readTextFile(`${dir}/${id}/${f}`);
        const meta = JSON.parse(await read("problem.json"));
        out.push({ id, title: meta.title, exports: meta.exports, spec: (await read("spec.md")).trim(),
            reference: await read("reference.js"), hidden: await read("hidden.test.js") });
    }
    if (filter) for (const f of filter) if (!out.some((p) => p.id === f)) throw new Error(`unknown problem ${f}`);
    return out;
}

function stubModule(exportsList) {
    const entries = exportsList.map((n) => `        ${n}: notImplemented(${JSON.stringify(n)}),`).join("\n");
    return `(function () {
    const notImplemented = (name) => function () { throw new Error(name + " is not implemented"); };
    return {
${entries}
    };
})()`;
}

// ---------------------------------------------------------------------------------------------
// sandbox

function sandboxCommand(opts) {
    const denoArgs = ["run", "--quiet", "--no-prompt", "--no-config", "--no-lock", "--v8-flags=--max-old-space-size=512", `${HERE}sandbox.js`];
    if (!opts.bwrap || opts.bwrap === "none") return { cmd: opts.deno, args: denoArgs };
    const bw = [
        "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        "--ro-bind", "/nix/store", "/nix/store",
        "--ro-bind", HERE, HERE,
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
        "--setenv", "HOME", "/tmp", "--setenv", "DENO_DIR", "/tmp/deno-dir", "--setenv", "NO_COLOR", "1",
        opts.deno, ...denoArgs,
    ];
    return { cmd: opts.bwrap, args: bw };
}

async function sandboxRun(opts, module, tests) {
    const { cmd, args } = sandboxCommand(opts);
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), opts.sandboxTimeoutMs);
    try {
        // The child inherits no environment from the harness.
        const child = new Deno.Command(cmd, {
            args, clearEnv: true, env: { NO_COLOR: "1", DENO_NO_UPDATE_CHECK: "1" },
            stdin: "piped", stdout: "piped", stderr: "piped", signal: ac.signal,
        }).spawn();
        const w = child.stdin.getWriter();
        await w.write(new TextEncoder().encode(JSON.stringify({ module, tests })));
        await w.close();
        const res = await child.output();
        const stdout = new TextDecoder().decode(res.stdout), stderr = new TextDecoder().decode(res.stderr);
        if (ac.signal.aborted) return { crash: `timed out after ${opts.sandboxTimeoutMs} ms` };
        try {
            return JSON.parse(stdout);
        } catch {
            return { crash: `sandbox exited with code ${res.code}: ${stderr.slice(-2000)}` };
        }
    } finally {
        clearTimeout(timer);
    }
}

function summarizeRun(r) {
    // -> { ok, passed, total, text }
    if (r.crash) return { ok: false, passed: 0, total: 0, text: `ERROR ${r.crash}` };
    if (!r.load.ok) return { ok: false, passed: 0, total: 0, text: `module failed to load: ${r.load.error}` };
    const s = r.suites[0];
    if (!s) return { ok: false, passed: 0, total: 0, text: "no test suite" };
    const lines = [];
    if (s.error) lines.push(`suite error: ${s.error}`);
    for (const x of s.tests) lines.push(x.ok ? `PASS ${x.name}` : `FAIL ${x.name}: ${x.error}`);
    const passed = s.tests.filter((x) => x.ok).length;
    lines.push(`${passed}/${s.tests.length} tests passed`);
    return { ok: s.ok, passed, total: s.tests.length, text: lines.join("\n") };
}

// ---------------------------------------------------------------------------------------------
// API client

function chatUrl(endpoint) {
    const base = endpoint.replace(/\/+$/, "");
    if (base.endsWith("/chat/completions")) return base;
    if (/\/v1$/.test(base)) return `${base}/chat/completions`;
    return `${base}/v1/chat/completions`;
}

async function chat(opts, messages, tools) {
    const body = { model: opts.model, messages, max_tokens: opts.maxTokens };
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    if (opts.topP !== undefined) body.top_p = opts.topP;
    if (opts.topK !== undefined) body.top_k = opts.topK;
    if (opts.reasoningEffort !== undefined) {
        // OpenAI-style field (llama-server) and the OpenRouter form.
        body.reasoning_effort = opts.reasoningEffort;
        body.reasoning = { effort: opts.reasoningEffort };
    }
    if (tools) { body.tools = tools; body.tool_choice = "auto"; }
    const headers = { "Content-Type": "application/json" };
    if (opts.token) headers.Authorization = `Bearer ${opts.token}`;
    // Network errors and 429/5xx (including 503 while a server loads its model) are retried with
    // backoff capped at 60 s for about 10 minutes, so a server restart does not fail attempts.
    let lastErr;
    for (let attempt = 0; attempt < 14; attempt++) {
        if (attempt) await new Promise((r) => setTimeout(r, Math.min(60000, 2000 * 2 ** attempt)));
        const t0 = performance.now();
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), opts.requestTimeoutS * 1000);
        try {
            const res = await fetch(chatUrl(opts.endpoint), { method: "POST", headers, body: JSON.stringify(body), signal: ac.signal });
            const text = await res.text();
            if (res.status === 429 || res.status >= 500) { lastErr = `HTTP ${res.status}: ${text.slice(0, 500)}`; continue; }
            if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 2000)}`);
            const json = JSON.parse(text);
            if (json.error) throw new Error(`API error: ${JSON.stringify(json.error).slice(0, 2000)}`);
            return { json, wallMs: performance.now() - t0 };
        } catch (e) {
            if (e.name === "AbortError") lastErr = `request timed out after ${opts.requestTimeoutS} s`;
            else if (e instanceof TypeError) lastErr = `network error: ${e.message}`;
            else throw e;
        } finally {
            clearTimeout(timer);
        }
    }
    throw new Error(`request failed after retries: ${lastErr}`);
}

function replyOf(json) {
    const choice = json.choices?.[0] ?? {};
    const m = choice.message ?? {};
    return {
        content: m.content ?? "",
        reasoning: m.reasoning_content ?? m.reasoning ?? "",
        toolCalls: m.tool_calls ?? [],
        finish: choice.finish_reason ?? null,
    };
}

// Stats over one attempt's responses.
function newStats() {
    return { requests: 0, wallMs: 0, promptTokens: 0, completionTokens: 0, predictedN: 0, predictedMs: 0, draftN: 0, draftAccepted: 0 };
}

function addStats(st, json, wallMs) {
    st.requests++;
    st.wallMs += wallMs;
    st.promptTokens += json.usage?.prompt_tokens ?? 0;
    st.completionTokens += json.usage?.completion_tokens ?? 0;
    const tm = json.timings;
    if (tm) {
        st.predictedN += tm.predicted_n ?? 0;
        st.predictedMs += tm.predicted_ms ?? 0;
        st.draftN += tm.draft_n ?? 0;
        st.draftAccepted += tm.draft_n_accepted ?? 0;
    }
}

// ---------------------------------------------------------------------------------------------
// prompts

const CONTRACT = `You write JavaScript as a single module expression: an immediately-invoked function expression
that takes no arguments and returns an object with the required exports, for example:

(function () {
    function helper(x) { return x * 2; }
    function double(x) { return helper(x); }
    return { double };
})()

The module runs in a restricted sandbox. There is no import, require or dynamic import(), no eval or Function,
no fetch, no timers, no filesystem and no globalThis. Use only standard built-ins (Math, Number, String, Array,
Map, Set, RegExp, JSON, errors such as TypeError, ...). Anything the module needs from outside is passed in
through the exported functions' parameters, as the specification describes.`;

const TEST_API = `A test suite is a function expression that receives the module's exports and a test API:

(function (mod, t) {
    t.test("doubles numbers", () => {
        t.equal(mod.double(2), 4);
    });
})

t.test(name, fn) registers a test. Assertions: t.ok(value, msg), t.equal(actual, expected, msg) (Object.is),
t.deepEqual(actual, expected, msg), t.throws(fn, msg). The suite gets nothing else: same sandbox rules as the module.`;

function plainPrompt(p) {
    return `${p.spec}\n\nRequired exports: ${p.exports.join(", ")}\n\nReply with the complete module in one \`\`\`js code block.`;
}

const AGENT_SYSTEM = `You are a careful software engineer working test-first.\n\n${CONTRACT}\n\n${TEST_API}

You work through tools. The tools act on this task's test suite and module only; you can not see or name files
and can not run commands. Work in the phase you are told to work in.`;

function redPrompt(p) {
    return `Specification:\n\n${p.spec}\n\nRequired exports: ${p.exports.join(", ")}

Phase 1 (RED): write a test suite for this specification with write_tests. Do not implement the module yet.
The current module is a placeholder whose exports all throw "not implemented". Use run_tests to check that
your tests run and fail against it. When the test suite is done, reply with a short summary and no tool call.`;
}

const GREEN_PROMPT = `Phase 2 (GREEN): implement the module with write_module so that your tests pass. You can read the
tests but not change them. Use run_tests to check your work. When all tests pass, reply with a short summary and
no tool call.`;

const REFACTOR_PROMPT = `Phase 3 (REFACTOR): improve the module's clarity and structure without changing its behaviour. The
tests must still pass; use run_tests after each change. If there is nothing worth improving, reply with a short
note and no tool call.`;

// ---------------------------------------------------------------------------------------------
// tools (capabilities on the current attempt's state)

const TOOL_DEFS = {
    read_spec: { description: "Return the specification of the task.", parameters: { type: "object", properties: {}, additionalProperties: false } },
    read_tests: { description: "Return the current test suite source.", parameters: { type: "object", properties: {}, additionalProperties: false } },
    write_tests: {
        description: "Replace the test suite with new source: a single function expression (function (mod, t) { ... }).",
        parameters: { type: "object", properties: { source: { type: "string", description: "complete test suite source" } }, required: ["source"], additionalProperties: false },
    },
    read_module: { description: "Return the current module source.", parameters: { type: "object", properties: {}, additionalProperties: false } },
    write_module: {
        description: "Replace the module with new source: a single IIFE (function () { ...; return { exports }; })().",
        parameters: { type: "object", properties: { source: { type: "string", description: "complete module source" } }, required: ["source"], additionalProperties: false },
    },
    run_tests: { description: "Run the current test suite against the current module and return the results.", parameters: { type: "object", properties: {}, additionalProperties: false } },
};

const PHASE_TOOLS = {
    red: ["read_spec", "read_tests", "write_tests", "run_tests"],
    green: ["read_spec", "read_tests", "read_module", "write_module", "run_tests"],
    refactor: ["read_spec", "read_tests", "read_module", "write_module", "run_tests"],
};

const MAX_SOURCE = 100_000;

function toolList(phase) {
    return PHASE_TOOLS[phase].map((name) => ({ type: "function", function: { name, ...TOOL_DEFS[name] } }));
}

async function runTool(opts, state, phase, name, rawArgs) {
    if (!PHASE_TOOLS[phase].includes(name)) return `error: ${name} is not available in the ${phase} phase`;
    let args = {};
    try {
        args = rawArgs ? JSON.parse(rawArgs) : {};
    } catch {
        return "error: arguments are not valid JSON";
    }
    const write = (kind, slot) => {
        const src = args.source;
        if (typeof src !== "string" || !src.trim()) return "error: source must be a non-empty string";
        if (src.length > MAX_SOURCE) return `error: source is longer than ${MAX_SOURCE} characters`;
        const c = checkSource(src, kind);
        if (!c.ok) return `rejected, nothing was saved:\n${c.errors.join("\n")}`;
        state[slot] = c.source;
        state.writes++;
        return `saved (${c.source.length} characters)`;
    };
    switch (name) {
        case "read_spec": return state.problem.spec;
        case "read_tests": return state.tests ?? "(no test suite yet)";
        case "read_module": return state.module;
        case "write_tests": return write("tests", "tests");
        case "write_module": return write("module", "module");
        case "run_tests": {
            if (!state.tests) return "error: there is no test suite yet";
            state.testRuns++;
            const s = summarizeRun(await sandboxRun(opts, state.module, [state.tests]));
            return s.text.length > 8000 ? s.text.slice(0, 8000) + "\n... (truncated)" : s.text;
        }
    }
    return `error: unknown tool ${name}`;
}

// ---------------------------------------------------------------------------------------------
// attempts

// Fenced code blocks, from fence lines. Fences are paired from the start and, if that leaves one unpaired, also from
// the end: a reply can begin inside a block (a provider split reasoning and content in the middle of one), and pairing
// only from the start would then take the prose between blocks for code. Sorted by where the block ends.
function codeBlocks(text) {
    const lines = text.split("\n");
    const fences = [];
    lines.forEach((l, i) => { const m = /^\s*```([A-Za-z0-9_+-]*)/.exec(l); if (m) fences.push({ i, lang: m[1].toLowerCase() }); });
    const pairs = new Map();
    const add = (o, c) => pairs.set(`${o.i}:${c.i}`, { lang: o.lang, end: c.i, code: lines.slice(o.i + 1, c.i).join("\n") + "\n" });
    for (let k = 0; k + 1 < fences.length; k += 2) add(fences[k], fences[k + 1]);
    if (fences.length % 2) for (let k = fences.length - 1; k >= 1; k -= 2) add(fences[k - 1], fences[k]);
    return [...pairs.values()].sort((a, b) => a.end - b.end);
}

function extractModule(text) {
    const js = codeBlocks(text).filter((b) => ["", "js", "javascript", "mjs"].includes(b.lang));
    for (const b of [...js].reverse()) if (checkSource(b.code, "module").ok) return { source: b.code, how: "last valid js block" };
    if (js.length) return { source: js[js.length - 1].code, how: "last js block (does not pass the checker)" };
    return { source: text, how: "whole reply (no code block)" };
}

async function plainAttempt(opts, p) {
    const stats = newStats();
    const messages = [{ role: "system", content: CONTRACT }, { role: "user", content: plainPrompt(p) }];
    const responses = [];
    const { json, wallMs } = await chat(opts, messages);
    addStats(stats, json, wallMs);
    responses.push(json);
    const r = replyOf(json);
    messages.push({ role: "assistant", content: r.content, reasoning: r.reasoning || undefined });
    const ex = extractModule(r.content);
    const check = checkSource(ex.source, "module");
    const hidden = summarizeRun(await sandboxRun(opts, check.ok ? check.source : ex.source, [p.hidden]));
    return {
        artifacts: { "module.js": check.ok ? check.source : ex.source },
        messages, responses, stats,
        result: { mode: "plain", finish: r.finish, extraction: ex.how, checker: check.errors, hidden: { ok: hidden.ok, passed: hidden.passed, total: hidden.total, report: hidden.text } },
    };
}

async function agenticAttempt(opts, p) {
    const stats = newStats();
    const state = { problem: p, tests: null, module: stubModule(p.exports), writes: 0, testRuns: 0 };
    const messages = [{ role: "system", content: AGENT_SYSTEM }];
    const responses = [], events = [];
    const phases = {};

    async function runPhase(phase, prompt, isDone, feedback) {
        messages.push({ role: "user", content: prompt });
        events.push({ phase, event: "start", at: messages.length - 1 });
        const info = { steps: 0, toolCalls: 0, nudges: 0, endedBy: null };
        let nudged = false;
        while (info.steps < opts.maxSteps) {
            info.steps++;
            const { json, wallMs } = await chat(opts, messages, toolList(phase));
            addStats(stats, json, wallMs);
            responses.push(json);
            const r = replyOf(json);
            const msg = { role: "assistant", content: r.content || "" };
            if (r.toolCalls.length) msg.tool_calls = r.toolCalls;
            messages.push(msg);
            if (r.reasoning) events.push({ phase, event: "reasoning", at: messages.length - 1, text: r.reasoning });
            if (r.toolCalls.length) {
                for (const tc of r.toolCalls) {
                    info.toolCalls++;
                    const out = await runTool(opts, state, phase, tc.function?.name, tc.function?.arguments);
                    messages.push({ role: "tool", tool_call_id: tc.id, content: out });
                }
                continue;
            }
            if (r.finish === "length") { info.endedBy = "length"; break; }
            const problem = await isDone();
            if (!problem) { info.endedBy = "done"; break; }
            if (nudged) { info.endedBy = "done-unmet"; break; }
            nudged = true;
            info.nudges++;
            messages.push({ role: "user", content: feedback(problem) });
        }
        if (!info.endedBy) info.endedBy = "step-limit";
        phases[phase] = info;
    }

    const runOwn = async () => summarizeRun(await sandboxRun(opts, state.module, [state.tests]));

    // RED: tests must exist and fail against the placeholder module.
    await runPhase("red", redPrompt(p), async () => {
        if (!state.tests) return "There is no test suite yet. Write it with write_tests.";
        const s = await runOwn();
        if (s.total === 0) return `The test suite does not define any tests:\n${s.text}`;
        if (s.ok) return "All tests pass against the placeholder module, so they do not test anything. Add tests that fail until the module is implemented.";
        return null;
    }, (problem) => `${problem}\nStay in phase 1 (RED).`);
    const redOwn = state.tests ? await runOwn() : null;
    const testsVsReference = state.tests ? summarizeRun(await sandboxRun(opts, p.reference, [state.tests])) : null;
    const redTests = state.tests;

    // GREEN: own tests pass.
    let greenModule = null;
    await runPhase("green", GREEN_PROMPT, async () => {
        if (!state.tests) return "There is no test suite.";
        const s = await runOwn();
        return s.ok ? null : `The tests do not all pass yet:\n${s.text.slice(0, 4000)}`;
    }, (problem) => `${problem}\nStay in phase 2 (GREEN).`);
    const greenOwn = state.tests ? await runOwn() : null;
    if (greenOwn?.ok) greenModule = state.module;
    const greenHidden = summarizeRun(await sandboxRun(opts, state.module, [p.hidden]));

    // REFACTOR: tests stay green; revert to the green module if they do not.
    let reverted = false;
    if (greenModule) {
        await runPhase("refactor", REFACTOR_PROMPT, async () => {
            const s = await runOwn();
            return s.ok ? null : `The tests fail after your changes:\n${s.text.slice(0, 4000)}`;
        }, (problem) => `${problem}\nFix the module; stay in phase 3 (REFACTOR).`);
        const after = await runOwn();
        if (!after.ok) {
            reverted = true;
            state.module = greenModule;
        }
    }
    const hidden = summarizeRun(await sandboxRun(opts, state.module, [p.hidden]));

    return {
        artifacts: { "tests.js": redTests ?? "", "module.js": state.module, ...(greenModule && greenModule !== state.module ? { "module.green.js": greenModule } : {}) },
        messages, responses, stats, events,
        result: {
            mode: "agentic", phases, writes: state.writes, testRuns: state.testRuns,
            red: redOwn && { failingAgainstPlaceholder: !redOwn.ok, tests: redOwn.total },
            testsVsReference: testsVsReference && { ok: testsVsReference.ok, passed: testsVsReference.passed, total: testsVsReference.total, report: testsVsReference.text },
            green: { ownTestsPass: !!greenOwn?.ok, hiddenPassed: greenHidden.passed, hiddenTotal: greenHidden.total },
            refactorReverted: reverted,
            hidden: { ok: hidden.ok, passed: hidden.passed, total: hidden.total, report: hidden.text },
        },
    };
}

// ---------------------------------------------------------------------------------------------
// output

function renderMarkdown(p, mode, run, att) {
    const lines = [`# ${p.id} (${p.title}), ${mode}, run ${run}`, ""];
    const ev = new Map((att.events ?? []).filter((e) => e.event === "start").map((e) => [e.at, e.phase]));
    const reasoning = new Map((att.events ?? []).filter((e) => e.event === "reasoning").map((e) => [e.at, e.text]));
    att.messages.forEach((m, i) => {
        if (ev.has(i)) lines.push(`---`, `## Phase: ${ev.get(i)}`, "");
        lines.push(`### ${m.role}${m.role === "tool" ? ` (${m.tool_call_id})` : ""}`, "");
        const r = m.reasoning ?? reasoning.get(i);
        if (r) lines.push("<details><summary>reasoning</summary>", "", "```text", r, "```", "</details>", "");
        if (m.content) lines.push(m.role === "tool" ? "```text\n" + m.content + "\n```" : m.content, "");
        for (const tc of m.tool_calls ?? []) {
            let args = tc.function?.arguments ?? "";
            try {
                const parsed = JSON.parse(args);
                if (typeof parsed.source === "string") {
                    lines.push(`**tool call** \`${tc.function.name}\` (${tc.id})`, "", "```js", parsed.source, "```", "");
                    continue;
                }
                args = JSON.stringify(parsed);
            } catch { /* keep raw */ }
            lines.push(`**tool call** \`${tc.function?.name}\`(${args}) (${tc.id})`, "");
        }
    });
    lines.push("---", "## Result", "", "```json", JSON.stringify(att.result, null, 2), "```", "");
    return lines.join("\n");
}

async function writeAttempt(dir, p, mode, run, att) {
    await Deno.mkdir(dir, { recursive: true });
    await Deno.writeTextFile(`${dir}/transcript.json`, JSON.stringify({ messages: att.messages, events: att.events ?? [], responses: att.responses }, null, 2));
    await Deno.writeTextFile(`${dir}/transcript.md`, renderMarkdown(p, mode, run, att));
    for (const [name, src] of Object.entries(att.artifacts)) await Deno.writeTextFile(`${dir}/${name}`, src);
    await Deno.writeTextFile(`${dir}/result.json`, JSON.stringify({ ...att.result, stats: att.stats }, null, 2));
}

function summaryMarkdown(meta, rows) {
    const lines = [`# Coding eval: ${meta.model}`, "", `endpoint: ${meta.endpoint}  `, `started: ${meta.started}  `, `runs per problem and mode: ${meta.runs}`, ""];
    const modes = [...new Set(rows.map((r) => r.mode))];
    for (const mode of modes) {
        const rs = rows.filter((r) => r.mode === mode);
        const solved = rs.filter((r) => r.ok).length;
        lines.push(`## ${mode}: ${solved}/${rs.length} attempts pass all hidden tests`, "", "| problem | attempts passing | hidden tests passed per attempt | gen t/s | notes |", "|---|---:|---|---:|---|");
        const ids = [...new Set(rs.map((r) => r.problem))];
        for (const id of ids) {
            const pr = rs.filter((r) => r.problem === id);
            const tps = pr.map((r) => r.tps).filter(Boolean);
            const notes = [...new Set(pr.map((r) => r.note).filter(Boolean))].join("; ");
            lines.push(`| ${id} | ${pr.filter((r) => r.ok).length}/${pr.length} | ${pr.map((r) => r.error ? "error" : `${r.passed}/${r.total}`).join(", ")} | ${tps.length ? (tps.reduce((a, b) => a + b, 0) / tps.length).toFixed(1) : ""} | ${notes} |`);
        }
        lines.push("");
    }
    return lines.join("\n");
}

// ---------------------------------------------------------------------------------------------
// main

async function selfCheck(opts, problems) {
    let bad = 0;
    for (const p of problems) {
        const ref = summarizeRun(await sandboxRun(opts, p.reference, [p.hidden]));
        const stub = summarizeRun(await sandboxRun(opts, stubModule(p.exports), [p.hidden]));
        // Tests that expect errors pass against the placeholder (it throws), so only require it to fail overall.
        const ok = ref.ok && !stub.ok;
        if (!ok) bad++;
        console.log(`${ok ? "ok  " : "FAIL"} ${p.id}: reference ${ref.passed}/${ref.total}, placeholder ${stub.passed}/${stub.total}`);
        if (!ref.ok) console.log(ref.text.split("\n").filter((l) => !l.startsWith("PASS")).map((l) => "     " + l).join("\n"));
    }
    return bad;
}

// Re-run extraction and grading on the saved replies of plain attempts, e.g. after a fix to either. Stats and the
// transcript stay as they were; module.js and result.json are rewritten, and the summary rebuilt.
async function regrade(opts, problems) {
    if (!opts.label) throw new Error("--regrade needs --label");
    const root = `${opts.out}/${opts.label}`;
    const meta = JSON.parse(await Deno.readTextFile(`${root}/meta.json`));
    const rows = [];
    let changed = 0;
    for (const p of problems) {
        for (const mode of ["plain", "agentic"]) {
            for (let run = 1; run <= (meta.runs ?? 3); run++) {
                const dir = `${root}/${p.id}/${mode}/run-${run}`;
                const prev = await readResult(dir);
                if (!prev) continue;
                if (mode === "agentic") { rows.push(rowFromResult(p.id, mode, run, prev)); continue; }
                const t = JSON.parse(await Deno.readTextFile(`${dir}/transcript.json`));
                const reply = t.messages.filter((m) => m.role === "assistant").at(-1)?.content ?? "";
                const ex = extractModule(reply);
                const check = checkSource(ex.source, "module");
                const src = check.ok ? check.source : ex.source;
                const hidden = summarizeRun(await sandboxRun(opts, src, [p.hidden]));
                const result = { ...prev, extraction: ex.how, checker: check.errors, hidden: { ok: hidden.ok, passed: hidden.passed, total: hidden.total, report: hidden.text } };
                if (prev.hidden.passed !== hidden.passed || prev.hidden.total !== hidden.total) {
                    changed++;
                    console.log(`${p.id} plain run ${run}: ${prev.hidden.passed}/${prev.hidden.total} -> ${hidden.passed}/${hidden.total}`);
                }
                await Deno.writeTextFile(`${dir}/module.js`, src);
                await Deno.writeTextFile(`${dir}/result.json`, JSON.stringify(result, null, 2));
                rows.push(rowFromResult(p.id, mode, run, result));
            }
        }
    }
    rows.sort((a, b) => a.problem.localeCompare(b.problem) || a.mode.localeCompare(b.mode) || a.run - b.run);
    await Deno.writeTextFile(`${root}/summary.json`, JSON.stringify({ meta, rows }, null, 2));
    await Deno.writeTextFile(`${root}/summary.md`, summaryMarkdown(meta, rows));
    console.log(`regraded ${root}: ${changed} plain attempts changed`);
    return 0;
}

function rowFromResult(problem, mode, run, r) {
    const st = r.stats ?? {};
    return {
        problem, mode, run, ok: r.hidden.ok, passed: r.hidden.passed, total: r.hidden.total,
        tps: st.predictedMs ? st.predictedN / (st.predictedMs / 1000) : null,
        note: mode === "agentic" && r.testsVsReference && !r.testsVsReference.ok ? "own tests reject the reference" : null,
    };
}

async function readResult(dir) {
    try {
        return JSON.parse(await Deno.readTextFile(`${dir}/result.json`));
    } catch {
        return null;
    }
}

async function main() {
    const opts = parseArgs(Deno.args);
    if (opts.help) { console.log(USAGE); return 0; }
    if (!opts.deno) throw new Error("--deno is required (run through run.sh)");
    const problems = await loadProblems(opts.problems);
    if (opts["self-check"]) return (await selfCheck(opts, problems)) ? 1 : 0;

    if (opts.regrade) return await regrade(opts, problems);
    if (!opts.endpoint || !opts.model) { console.error(USAGE); return 2; }
    opts.token ??= Deno.env.get("CODING_EVAL_TOKEN") || undefined;
    const started = new Date().toISOString();
    const label = opts.label ?? `${opts.model.replace(/[^A-Za-z0-9._-]+/g, "_")}_${started.replace(/[:.]/g, "-")}`;
    const root = `${opts.out}/${label}`;
    await Deno.mkdir(root, { recursive: true });
    const meta = {
        endpoint: opts.endpoint, model: opts.model, started, runs: opts.runs, mode: opts.mode,
        temperature: opts.temperature ?? null, topP: opts.topP ?? null, topK: opts.topK ?? null, reasoningEffort: opts.reasoningEffort ?? null, maxTokens: opts.maxTokens, maxSteps: opts.maxSteps,
        problems: problems.map((p) => p.id), sandbox: opts.bwrap && opts.bwrap !== "none" ? "deno (no permissions) under bubblewrap" : "deno (no permissions)",
    };
    await Deno.writeTextFile(`${root}/meta.json`, JSON.stringify(meta, null, 2));

    const modes = opts.mode === "both" ? ["plain", "agentic"] : [opts.mode];
    const jobs = [];
    for (const p of problems) for (const mode of modes) for (let run = 1; run <= opts.runs; run++) jobs.push({ p, mode, run });

    const rows = [];
    const writeSummary = async () => {
        rows.sort((a, b) => a.problem.localeCompare(b.problem) || a.mode.localeCompare(b.mode) || a.run - b.run);
        await Deno.writeTextFile(`${root}/summary.json`, JSON.stringify({ meta, rows }, null, 2));
        await Deno.writeTextFile(`${root}/summary.md`, summaryMarkdown(meta, rows));
    };
    const todo = [];
    for (const job of jobs) {
        const prev = opts.resume ? await readResult(`${root}/${job.p.id}/${job.mode}/run-${job.run}`) : null;
        if (prev) rows.push(rowFromResult(job.p.id, job.mode, job.run, prev));
        else todo.push(job);
    }
    if (opts.resume) console.log(`resume: ${rows.length} attempts already done, ${todo.length} to run`);
    await writeSummary();

    let next = 0;
    async function worker() {
        while (next < todo.length) {
            const { p, mode, run } = todo[next++];
            const dir = `${root}/${p.id}/${mode}/run-${run}`;
            const t0 = performance.now();
            let row;
            try {
                const att = mode === "plain" ? await plainAttempt(opts, p) : await agenticAttempt(opts, p);
                await writeAttempt(dir, p, mode, run, att);
                row = rowFromResult(p.id, mode, run, { ...att.result, stats: att.stats });
            } catch (e) {
                await Deno.mkdir(dir, { recursive: true });
                await Deno.writeTextFile(`${dir}/error.txt`, String(e?.stack ?? e));
                row = { problem: p.id, mode, run, ok: false, passed: 0, total: 0, error: String(e?.message ?? e) };
            }
            rows.push(row);
            const secs = ((performance.now() - t0) / 1000).toFixed(0);
            console.log(`${row.ok ? "PASS" : row.error ? "ERR " : "fail"} ${p.id} ${mode} run ${run}: ${row.error ? row.error.slice(0, 200) : `${row.passed}/${row.total}`} (${secs} s)`);
            await writeSummary();
        }
    }
    await Promise.all(Array.from({ length: Math.max(1, opts.parallel) }, worker));
    console.log(`results in ${root}`);
    return 0;
}

Deno.exit(await main());
