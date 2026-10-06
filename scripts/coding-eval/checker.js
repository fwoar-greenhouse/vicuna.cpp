// Static checks for model-written sources, done on the parsed AST (not with regexes).
//
// - The source must be exactly one expression:
//     module: an IIFE `(function () { ... })()` or `(() => { ... })()` with no arguments
//     tests:  a function expression `(function (mod, t) { ... })` with at most 2 parameters
// - Dynamic `import(...)` and `import.meta` are rejected anywhere.
// - References to `eval` and `Function` are rejected (they are also unavailable at run time).

import * as acorn from "./vendor/acorn.mjs";

const BANNED_IDENTIFIERS = new Set(["eval", "Function"]);

export function normalizeSource(src) {
    // Allow a trailing `;` after the expression; nothing else is changed.
    return String(src).trim().replace(/;+\s*$/, "");
}

function isFunctionNode(node) {
    return node && (node.type === "FunctionExpression" || node.type === "ArrowFunctionExpression");
}

// True if `node` (an Identifier) is used as a variable reference, not as a property name or key.
function isReference(node, parent, key) {
    if (!parent) return true;
    if (parent.type === "MemberExpression" && key === "property" && !parent.computed) return false;
    if ((parent.type === "Property" || parent.type === "MethodDefinition" || parent.type === "PropertyDefinition") &&
        key === "key" && !parent.computed) return false;
    if (parent.type === "LabeledStatement" || parent.type === "BreakStatement" || parent.type === "ContinueStatement") return false;
    return true;
}

function walk(node, visit, parent = null, key = null) {
    if (!node || typeof node.type !== "string") return;
    visit(node, parent, key);
    for (const k of Object.keys(node)) {
        if (k === "loc" || k === "start" || k === "end" || k === "range") continue;
        const v = node[k];
        if (Array.isArray(v)) {
            for (const child of v) walk(child, visit, node, k);
        } else if (v && typeof v.type === "string") {
            walk(v, visit, node, k);
        }
    }
}

function lineOf(src, pos) {
    // The source is parsed with one extra line in front, see checkSource.
    return src.slice(0, pos).split("\n").length - 1;
}

export function checkSource(rawSrc, kind) {
    const src = normalizeSource(rawSrc);
    const wrapped = `(\n${src}\n)`;
    const errors = [];
    let ast;
    try {
        ast = acorn.parse(wrapped, { ecmaVersion: "latest", sourceType: "script", locations: true });
    } catch (e) {
        const line = e.loc ? Math.max(1, e.loc.line - 1) : "?";
        return { ok: false, source: src, errors: [`syntax error at line ${line}: ${e.message.replace(/ \(\d+:\d+\)$/, "")}`] };
    }

    if (ast.body.length !== 1 || ast.body[0].type !== "ExpressionStatement") {
        errors.push("source must be a single expression");
    } else {
        let expr = ast.body[0].expression;
        if (kind === "module") {
            const ok = expr.type === "CallExpression" && isFunctionNode(expr.callee) && expr.arguments.length === 0;
            if (!ok) errors.push("module must be an IIFE with no arguments: (function () { ...; return { exports }; })()");
        } else if (kind === "tests") {
            const ok = isFunctionNode(expr) && expr.params.length <= 2;
            if (!ok) errors.push("tests must be a function expression taking (mod, t): (function (mod, t) { t.test(...); })");
        }
    }

    walk(ast, (node, parent, key) => {
        const line = lineOf(wrapped, node.start);
        if (node.type === "ImportExpression") {
            errors.push(`line ${line}: dynamic import(...) is not allowed`);
        } else if (node.type === "MetaProperty" && node.meta.name === "import") {
            errors.push(`line ${line}: import.meta is not allowed`);
        } else if (node.type === "Identifier" && BANNED_IDENTIFIERS.has(node.name) && isReference(node, parent, key)) {
            errors.push(`line ${line}: ${node.name} is not available`);
        }
    });

    return { ok: errors.length === 0, source: src, errors };
}
