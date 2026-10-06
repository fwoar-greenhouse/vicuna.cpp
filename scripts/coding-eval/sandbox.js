// Evaluator for model-written code. Runs as its own `deno run` process with no
// permission flags (no fs, net, env or subprocess access), usually under bubblewrap.
//
// Input (stdin, JSON): { module: "<IIFE source>", tests: ["<function source>", ...] }
// Output (stdout, JSON): { load: {ok, error}, suites: [{ok, error, tests: [{name, ok, error}]}] }
//
// A module is an expression `(function () { ...; return { exports }; })()`.
// A test suite is an expression `(function (mod, t) { t.test("name", () => ...); })`.
// Each gets only what is passed in. Layers:
//   1. checker.js rejects dynamic import(), import.meta, eval and Function on the AST
//   2. ambient globals are shadowed in the compiled function scope
//   3. after compiling, the constructors of all function kinds are poisoned,
//      so code can not build new code at run time
//   4. Deno permissions (none) and bubblewrap (no network, empty filesystem)

import { checkSource } from "./checker.js";

const STDOUT = Deno.stdout;

const SHADOWED = [
    "Deno", "fetch", "globalThis", "self", "window", "WebSocket", "Worker", "SharedWorker",
    "XMLHttpRequest", "navigator", "location", "importScripts", "caches", "localStorage",
    "sessionStorage", "BroadcastChannel", "EventSource", "WebAssembly",
    "Function", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
];
// `eval` can not be a parameter name in strict mode. checker.js rejects every reference
// to it, and the global object (globalThis, self, window) is shadowed above.

function compile(src) {
    const body = `"use strict";\nreturn (\n${src}\n);`;
    return new Function(...SHADOWED, body);
}

function run(compiled) {
    return compiled(...SHADOWED.map(() => undefined));
}

function poisonConstructors() {
    const kinds = [
        function () {},
        async function () {},
        function* () {},
        async function* () {},
    ];
    const disabled = function () {
        throw new TypeError("creating functions from strings is disabled");
    };
    for (const f of kinds) {
        Object.defineProperty(Object.getPrototypeOf(f), "constructor", {
            value: disabled, writable: false, enumerable: false, configurable: false,
        });
    }
}

function fmtError(e) {
    if (e instanceof Error) {
        return `${e.name}: ${e.message}`;
    }
    try {
        return `thrown: ${JSON.stringify(e)}`;
    } catch {
        return `thrown: ${String(e)}`;
    }
}

function deepEqual(a, b) {
    if (Object.is(a, b)) return true;
    if (typeof a !== typeof b || a === null || b === null || typeof a !== "object") return false;
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual(a[k], b[k]));
}

function show(v) {
    try {
        return JSON.stringify(v);
    } catch {
        return String(v);
    }
}

// The only capability given to test suites besides the module object.
function makeTestApi(collected) {
    const fail = (msg) => {
        throw new Error(msg);
    };
    return Object.freeze({
        test(name, fn) {
            collected.push({ name: String(name), fn });
        },
        ok(v, msg) {
            if (!v) fail(msg ?? `expected truthy, got ${show(v)}`);
        },
        equal(actual, expected, msg) {
            if (!Object.is(actual, expected)) fail(msg ?? `expected ${show(expected)}, got ${show(actual)}`);
        },
        deepEqual(actual, expected, msg) {
            if (!deepEqual(actual, expected)) fail(msg ?? `expected ${show(expected)}, got ${show(actual)}`);
        },
        throws(fn, msg) {
            let threw = false;
            try {
                fn();
            } catch {
                threw = true;
            }
            if (!threw) fail(msg ?? "expected function to throw");
        },
    });
}

async function runSuite(compiled, mod) {
    const collected = [];
    const t = makeTestApi(collected);
    try {
        const suiteFn = run(compiled);
        if (typeof suiteFn !== "function") {
            return { ok: false, error: "test source must be a function expression taking (mod, t)", tests: [] };
        }
        await suiteFn(mod, t);
    } catch (e) {
        return { ok: false, error: `while defining tests: ${fmtError(e)}`, tests: [] };
    }
    const tests = [];
    for (const { name, fn } of collected) {
        try {
            await fn();
            tests.push({ name, ok: true });
        } catch (e) {
            tests.push({ name, ok: false, error: fmtError(e) });
        }
    }
    return { ok: tests.length > 0 && tests.every((x) => x.ok), error: tests.length ? null : "no tests defined", tests };
}

async function main() {
    const input = JSON.parse(await new Response(Deno.stdin.readable).text());
    const testSources = input.tests ?? [];
    const out = { load: { ok: true, error: null }, suites: [] };

    // Check and compile everything before any model-written code runs.
    const modCheck = checkSource(input.module ?? "", "module");
    const suiteChecks = testSources.map((s) => checkSource(s, "tests"));
    let modCompiled = null;
    if (!modCheck.ok) {
        out.load = { ok: false, error: `rejected: ${modCheck.errors.join("; ")}` };
    } else {
        try {
            modCompiled = compile(modCheck.source);
        } catch (e) {
            out.load = { ok: false, error: fmtError(e) };
        }
    }
    const suiteCompiled = suiteChecks.map((c) => {
        if (!c.ok) return { error: `rejected: ${c.errors.join("; ")}` };
        try {
            return { fn: compile(c.source) };
        } catch (e) {
            return { error: fmtError(e) };
        }
    });
    poisonConstructors();

    let mod;
    if (out.load.ok) {
        try {
            mod = run(modCompiled);
            if (mod === null || typeof mod !== "object") {
                throw new TypeError("module expression must evaluate to an object of exports");
            }
        } catch (e) {
            out.load = { ok: false, error: fmtError(e) };
        }
    }
    for (const s of suiteCompiled) {
        if (s.error) out.suites.push({ ok: false, error: s.error, tests: [] });
        else if (!out.load.ok) out.suites.push({ ok: false, error: "module failed to load", tests: [] });
        else out.suites.push(await runSuite(s.fn, mod));
    }
    await STDOUT.write(new TextEncoder().encode(JSON.stringify(out)));
}

await main();
