#!/usr/bin/env python3
"""Static rule lint for an okc submission file. Catches what the judge's static checks and the
competition rules forbid, before a daily run is spent on it. It cannot prove compliance (no GPU
here); it refuses the obvious violations and flags the suspicious ones.

usage: okc_lint.py FILE [--problem NAME] [--json]
exit 0 clean (warnings allowed), 1 findings at level error, 2 usage.

Rule ids (error unless marked warn):
  marker        first line must be `# okc: problem=<name>` (and match --problem when given)
  entry         kernel_fn must be defined; prepare() for problems that have weights
  host-sync     .item( .cpu( .numpy( .tolist( torch.cuda.synchronize( stream.synchronize(
  process       subprocess / os.system / os.popen / multiprocessing / os.fork / exec( / eval(
  network       socket / requests / urllib / http.client / ssl
  harness       import of reference/checks/baseline, sys.modules edits, inspect.stack/currentframe,
                reading harness paths, monkeypatching torch.cuda.Event/graph/timing
  dtype         float8_e5m2, int8/uint8/float16 casts of activations (warn: review by hand)
  cache-data    module-level dict/list keyed or filled inside kernel_fn with tensors from `data`
                (warn: caching derived-from-data is forbidden; config-derived is fine)
  print-tensor  print( inside kernel_fn whose argument looks like a tensor (warn)
"""
import argparse
import ast
import json
import re
import sys

PROBLEMS_WITH_WEIGHTS = {"muse_bf16_proj", "muse_decode_layer", "muse_moe_w4a16", "muse_prefill_layer"}

HOST_SYNC = re.compile(r"\.(item|cpu|numpy|tolist)\s*\(|torch\.cuda\.synchronize\s*\(|\.synchronize\s*\(")
PROCESS = re.compile(r"\b(subprocess|multiprocessing)\b|os\.(system|popen|fork|exec[lv]p?e?)\s*\(|(?<![\w.])(exec|eval)\s*\(")
NETWORK = re.compile(r"\b(socket|requests|urllib|http\.client|ssl|aiohttp|httpx)\b")
HARNESS_IMPORT = re.compile(r"^\s*(from\s+(reference|checks|baseline|harness|eval)\b|import\s+(reference|checks|baseline|harness)\b)", re.M)
HARNESS_OTHER = re.compile(r"sys\.modules\s*\[|inspect\.(stack|currentframe|getouterframes)|torch\.cuda\.(Event|CUDAGraph|graph)\s*=|time\.(perf_counter|time)\s*=|/harness/|/challenge/|__builtins__")
DTYPE = re.compile(r"float8_e5m2|\.to\(\s*torch\.(int8|uint8|float16|half)\s*\)|dtype\s*=\s*torch\.(int8|uint8|float16|half)\b")
MARKER = re.compile(r"^#\s*okc:\s*problem=(\S+)\s*$")


def find_kernel_fn(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "kernel_fn":
            return node
    return None


def timed_line_ranges(tree):
    """Line ranges of kernel_fn and every function in this file it can reach by simple-name calls."""
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "kernel_fn" not in funcs:
        return []
    seen, todo = set(), ["kernel_fn"]
    while todo:
        name = todo.pop()
        if name in seen or name not in funcs:
            continue
        seen.add(name)
        for n in ast.walk(funcs[name]):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name):
                    todo.append(n.func.id)
                elif isinstance(n.func, ast.Attribute):
                    todo.append(n.func.attr)
    return [(funcs[n].lineno, funcs[n].end_lineno or funcs[n].lineno) for n in seen]


def lint(path, problem=None):
    src = open(path, encoding="utf-8", errors="replace").read()
    lines = src.splitlines()
    findings = []

    def add(rule, level, line, msg):
        findings.append({"rule": rule, "level": level, "line": line, "msg": msg})

    m = MARKER.match(lines[0] if lines else "")
    if not m:
        add("marker", "error", 1, "first line is not `# okc: problem=<name>`")
    elif problem and m.group(1) != problem:
        add("marker", "error", 1, f"marker names {m.group(1)}, expected {problem}")
    marker_problem = m.group(1) if m else problem

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        add("syntax", "error", e.lineno or 0, f"syntax error: {e.msg}")
        return findings
    kfn = find_kernel_fn(tree)
    if kfn is None:
        add("entry", "error", 0, "kernel_fn is not defined")
    if marker_problem in PROBLEMS_WITH_WEIGHTS and not any(
        isinstance(n, ast.FunctionDef) and n.name == "prepare" for n in ast.walk(tree)
    ):
        add("entry", "error", 0, "prepare(weights) is not defined but this problem has weights")

    timed = timed_line_ranges(tree)   # lines reachable from kernel_fn; prepare() is untimed

    def in_timed(i):
        return any(a <= i <= b for a, b in timed)

    for i, line in enumerate(lines, 1):
        code = line.split("#", 1)[0]
        if HOST_SYNC.search(code):
            if in_timed(i):
                add("host-sync", "error", i, code.strip())
            elif kfn is not None:
                add("host-sync", "warn", i, code.strip() + "  (outside kernel_fn's call graph; fine in prepare())")
        if PROCESS.search(code):
            add("process", "error", i, code.strip())
        if NETWORK.search(code):
            add("network", "error", i, code.strip())
        if HARNESS_OTHER.search(code):
            add("harness", "error", i, code.strip())
        if DTYPE.search(code):
            add("dtype", "warn", i, code.strip() + "  (allowed only where SGLang does the same)")
    for mm in HARNESS_IMPORT.finditer(src):
        add("harness", "error", src[: mm.start()].count("\n") + 1, mm.group(0).strip())

    if kfn is not None:
        module_containers = {
            n.targets[0].id
            for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, (ast.Dict, ast.List, ast.Call))
        }
        for node in ast.walk(kfn):
            # module-level container written inside kernel_fn: x[...] = ... or x.setdefault / append
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id in module_containers:
                add("cache-data", "warn", node.lineno, f"writes module-level {node.value.id} inside kernel_fn; must not depend on `data`")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("setdefault", "append", "update") \
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in module_containers:
                add("cache-data", "warn", node.lineno, f"mutates module-level {node.func.value.id} inside kernel_fn; must not depend on `data`")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                txt = ast.get_source_segment(src, node) or ""
                if re.search(r"\bdata\[|out\b|tensor|\.shape\b", txt) and ".shape" not in txt:
                    add("print-tensor", "warn", node.lineno, txt.strip()[:120])
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--problem")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    findings = lint(a.file, a.problem)
    errors = [f for f in findings if f["level"] == "error"]
    if a.json:
        print(json.dumps({"file": a.file, "ok": not errors, "findings": findings}))
    else:
        for f in findings:
            print(f"{f['level']:5} {f['rule']:12} L{f['line']}: {f['msg']}")
        print(("CLEAN" if not findings else ("OK with warnings" if not errors else "REJECT")) + f": {a.file}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
