#!/usr/bin/env python3
"""Lines-of-code per kernel for muse_decode_layer/submission.py across every commit that touched it.

Writes data/loc.json (schema in the campaign brief). Python 3 stdlib only; reads git objects only.

Method
------
* CUDA sources = every Python string constant (found with `ast`, so escaped single-line strings are
  unescaped for free) that contains __global__/__device__/#include/at::Tensor.  FLUSH_* (L2-flush
  probe kernels) are excluded from the kernel table.
* Runtime-derived sources are reconstructed by executing, in a sandbox namespace (no torch, no I/O
  modules), the file's top-level assignments and plain function definitions, then calling every
  zero-argument `*_source()` function (_de_source, _gle_source, ...) and reading back dict entries
  added at import time (_MA_ALT_SRCS["moe_helper0df"] = _df_patch(...)).  The kernel renames done by
  the matching *_ext() (`.replace('sk2_kernel', 'sk2_kernel_gle' + tag)`) are applied without the
  hash, giving the bare profiler base (sk2_kernel_gle, sk2_kernel_narrow).
* Comments and blank lines are stripped for `own`/`helpers`/`total`; `raw_total` keeps them.
* Preprocessor: -D sets are chosen per build variant at runtime, so ALL #if/#else branches are
  counted (noted per kernel as the number of #if lines inside its closure).
* __global__ / __device__ functions are found by brace matching; the call graph is identifier
  intersection with known __device__ names (same source first, else any source in the commit).
  Overloads / template specializations of one name are all included.
* Triton: @triton.jit functions; a jit function is a kernel if it is launched (`name[`) or not
  called by another jit function; its helpers are the jit functions it reaches transitively.
* When one kernel base is defined in several sources (moe_fused in MOE_CUDA_SRC and moe_helper0/df),
  the largest total is reported and the others are listed in `variants`.
"""
import ast, json, os, re, subprocess, sys, hashlib, math, itertools, functools, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--show-toplevel"], text=True).strip()
PATH = "muse_decode_layer/submission.py"
CUDA_MARK = re.compile(r"__global__|__device__|#\s*include|at::Tensor|torch::Tensor")
EXCLUDE_SRC = re.compile(r"^FLUSH")
FN_RE = re.compile(r"[;{=]|\b(\w+)\s*(?:<[^<>;{}()]*>)?\s*\(")  # name, optional <explicit specialization args>, (
SKIP_CALL_WORDS = {"__launch_bounds__", "__align__", "alignas", "__declspec", "decltype", "sizeof"}


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True)


# ----------------------------------------------------------------------------- C/CUDA text helpers
def strip_comments(s):
    """Blank out // and /* */ comments and string/char literal contents; length and newlines preserved."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            j = s.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", s[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c and s[j] != "\n":
                j += 2 if s[j] == "\\" else 1
            seg = s[i:j + 1]
            out.append(c + re.sub(r"[^\n]", "x", seg[1:-1]) + (seg[-1] if len(seg) > 1 else ""))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def match(s, i, o, c):
    """s[i] == o; return index of matching c."""
    d = 0
    for j in range(i, len(s)):
        if s[j] == o:
            d += 1
        elif s[j] == c:
            d -= 1
            if d == 0:
                return j
    return -1


def code_lines(text):
    return sum(1 for l in text.split("\n") if l.strip())


def find_functions(code, qual):
    """Yield (name, start, end) for function definitions introduced by `qual` in comment-stripped code."""
    for m in re.finditer(r"\b" + qual + r"\b", code):
        i = m.end()
        name = None
        while True:
            w = FN_RE.search(code, i)
            if not w or w.group(1) is None:
                break  # variable / declaration without parameter list
            if w.group(1) in SKIP_CALL_WORDS:
                i = match(code, w.end() - 1, "(", ")") + 1
                if i <= 0:
                    break
                continue
            name = w.group(1)
            p = w.end() - 1
            break
        if not name or name in ("if", "for", "while", "switch", "return"):
            continue
        q = match(code, p, "(", ")")
        if q < 0:
            continue
        k = q + 1
        while k < len(code) and code[k] not in "{;":
            k += 1
        if k >= len(code) or code[k] == ";":
            continue  # prototype
        if re.search(r"[=]", code[q + 1:k]) and "->" not in code[q + 1:k]:
            continue
        e = match(code, k, "{", "}")
        if e < 0:
            continue
        # start at the line beginning; include an immediately preceding template<...> line
        ls = code.rfind("\n", 0, m.start()) + 1
        prev = code.rfind("\n", 0, max(ls - 1, 0)) + 1
        if ls > 0 and re.match(r"\s*template\s*<", code[prev:ls]) and code[prev:ls].strip().endswith(">"):
            ls = prev
        yield name, ls, e + 1


def analyse_cuda_source(srcname, text):
    code = strip_comments(text)
    funcs = {"global": [], "device": []}
    seen = set()
    for kind, qual in (("global", "__global__"), ("device", "__device__")):
        for name, s, e in find_functions(code, qual):
            if (s, e) in seen:
                continue
            seen.add((s, e))
            body = code[s:e]
            funcs[kind].append({
                "name": name, "src": srcname,
                "code": code_lines(body), "raw": text.count("\n", s, e) + 1,
                "ifs": len(re.findall(r"^\s*#\s*if", body, re.M)),
                "idents": set(re.findall(r"\b[A-Za-z_]\w*\b", body)) - {name},
            })
    return funcs


# ----------------------------------------------------------------------------- python file helpers
def owner_name(node, parents):
    p = node
    while p in parents:
        q = parents[p]
        if isinstance(q, (ast.Assign, ast.AnnAssign)):
            tg = q.targets[0] if isinstance(q, ast.Assign) else q.target
            return ast.unparse(tg)
        if isinstance(q, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "fn:" + q.name
        p = q
    return "?"


def is_jit(fn):
    for d in fn.decorator_list:
        t = ast.unparse(d)
        # triton.jit, a bare jit, or an alias such as `_glx.jit` / `_glx.jit(...)`
        if re.search(r"\bjit\b", t) and "triton" in t or t in ("jit", "triton.jit") \
                or re.fullmatch(r"\w+\.jit(\(.*\))?", t, flags=re.S):
            return True
    return False


def sandbox_sources(tree, src_text):
    """Execute top-level assignments/defs in a restricted namespace to recover runtime-built sources."""
    import builtins, types
    fake_os = types.SimpleNamespace(environ={}, path=os.path, getenv=lambda k, d=None: d)  # env unset -> build defaults
    ns = {"os": fake_os, "__builtins__": {k: getattr(builtins, k) for k in dir(builtins)
                           if k not in ("open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "exit", "quit")},
          "hashlib": hashlib, "_hashlib": hashlib, "re": re, "math": math, "itertools": itertools,
          "functools": functools, "textwrap": textwrap}
    funcs = {}
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef,)):
            if st.decorator_list:
                continue
            funcs[st.name] = st
            mod = ast.Module(body=[st], type_ignores=[])
        elif isinstance(st, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            mod = ast.Module(body=[st], type_ignores=[])
        else:
            continue
        try:
            exec(compile(mod, "<sub>", "exec"), ns)
        except Exception:
            pass
    out = []
    # dict entries holding CUDA (e.g. _MA_ALT_SRCS["moe_helper0df"])
    for k, v in list(ns.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str) and CUDA_MARK.search(vv) and "__global__" in vv:
                    out.append((f"{k}[{kk!r}]", vv, {}))
    # zero-arg *_source() builders + renames applied by their callers
    for name, fn in funcs.items():
        if not name.endswith("_source") or fn.args.args or fn.args.vararg:
            continue
        try:
            s = ns[name]()
        except Exception:
            continue
        if not isinstance(s, str) or "__global__" not in s:
            continue
        ren = {}
        for caller in funcs.values():
            seg = ast.get_source_segment(src_text, caller) or ""
            if re.search(r"\b" + name + r"\s*\(", seg) and caller.name != name:
                for a, b in re.findall(r"replace\(\s*['\"](\w+)['\"]\s*,\s*['\"](\w+?)_?['\"]\s*\+\s*tag\s*\)", seg):
                    ren[a] = b
        out.append((f"fn:{name}()", s, ren))
    return out


def analyse_commit(c):
    text = git("show", f"{c}:{PATH}")
    lines = text.split("\n")
    res = {"file": {"lines": len(lines) - (1 if text.endswith("\n") else 0)}, "kernels": {}}
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        res["error"] = f"syntax: {e}"
        return res
    parents = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}

    # ---- CUDA constants
    cuda_consts, file_span = [], set()
    cuda_raw = cuda_code = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and CUDA_MARK.search(n.value) and n.value.count("\n") >= 2:
            nm = owner_name(n, parents)
            cuda_consts.append((nm, n.value, {}))
            file_span.update(range(n.lineno, n.end_lineno + 1))
            cuda_raw += n.value.count("\n") + 1
            cuda_code += code_lines(strip_comments(n.value))
    derived = []
    try:
        derived = sandbox_sources(tree, text)
    except Exception as e:  # pragma: no cover
        res["sandbox_error"] = repr(e)

    # ---- triton
    jit = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_jit(n):
            seg_lines = lines[n.lineno - 1:n.end_lineno]
            doc = set()
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                doc = set(range(n.body[0].lineno, n.body[0].end_lineno + 1))
            code = sum(1 for i, l in enumerate(seg_lines, n.lineno)
                       if l.strip() and not l.strip().startswith("#") and i not in doc)
            calls = {ast.unparse(x.func).split(".")[-1] for x in ast.walk(n) if isinstance(x, ast.Call)}
            jit[n.name] = {"code": code, "raw": len(seg_lines), "calls": calls,
                           "span": range(n.lineno, n.end_lineno + 1), "dec": n.decorator_list[0].lineno}
    tri_span = set()
    for j in jit.values():
        tri_span.update(range(j["dec"], j["span"].stop))
    tri_span -= file_span
    triton_code = sum(j["code"] for j in jit.values())
    py_lines = [i for i in range(1, res["file"]["lines"] + 1) if i not in file_span and i not in tri_span]
    res["file"].update({
        "cuda": cuda_raw, "triton": len(tri_span), "python": len(py_lines),
        "cuda_code": cuda_code, "triton_code": triton_code,
        "python_code": sum(1 for i in py_lines if lines[i - 1].strip() and not lines[i - 1].strip().startswith("#")),
        "cuda_file_lines": len(file_span),
        "n_cuda_sources": len(cuda_consts) + len(derived),
    })

    # ---- CUDA kernels
    per_src = {}
    for nm, s, ren in cuda_consts + derived:
        f = analyse_cuda_source(nm, s)
        for g in f["global"]:
            g["base"] = ren.get(g["name"], g["name"])
        per_src[nm] = f
    dev_by_src = {nm: {} for nm in per_src}
    dev_any = {}
    for nm, f in per_src.items():
        for d in f["device"]:
            dev_by_src[nm].setdefault(d["name"], []).append(d)
            dev_any.setdefault(d["name"], {}).setdefault(nm, []).append(d)

    def unit(nm):  # sources that share a name prefix form one translation unit (bfp_*, _WG_SRC_HEAD/BODY, ...)
        return re.match(r"(?:fn:)?_*([A-Za-z0-9]+)", nm).group(1)

    def resolve(name, src):
        if name in dev_by_src[src]:
            return dev_by_src[src][name]
        for other, lst in dev_any.get(name, {}).items():
            if unit(other) == unit(src) and not other.startswith("fn:"):
                return lst
        return []

    cands = {}
    for nm, f in per_src.items():
        if EXCLUDE_SRC.search(nm):
            continue
        for g in f["global"]:
            seen, stack, helpers = set(), list(g["idents"] & dev_any.keys()), []
            while stack:
                h = stack.pop()
                if h in seen:
                    continue
                seen.add(h)
                for d in resolve(h, nm):
                    helpers.append(d)
                    stack.extend((d["idents"] & dev_any.keys()) - seen)
            hc = sum(d["code"] for d in helpers)
            hr = sum(d["raw"] for d in helpers)
            ifs = g["ifs"] + sum(d["ifs"] for d in helpers)
            cands.setdefault(g["base"], []).append({
                "lang": "cuda", "own": g["code"], "helpers": hc, "total": g["code"] + hc,
                "raw_total": g["raw"] + hr, "helpers_list": sorted(seen),
                "source_const": nm, "note": (f"{ifs} #if lines in closure; all branches counted" if ifs else ""),
            })
    # same-source duplicates (e.g. #if/#else alternative bodies) -> keep largest; cross-source -> variants
    for base, lst in cands.items():
        lst.sort(key=lambda r: -r["total"])
        best = dict(lst[0])
        others = {}
        for r in lst[1:]:
            others.setdefault(r["source_const"], r["total"])
        if others:
            best["variants"] = others
            best["note"] = (best["note"] + "; " if best["note"] else "") + \
                f"also defined in {len(others)} other source(s)/body(ies); largest reported"
        res["kernels"][base] = best

    # ---- Triton kernels
    called = set().union(*[j["calls"] for j in jit.values()]) if jit else set()
    for name, j in jit.items():
        launched = re.search(r"\b" + re.escape(name) + r"\s*\[", text) is not None
        if not launched and name in called:
            continue
        seen, stack = set(), list(j["calls"] & jit.keys() - {name})
        while stack:
            h = stack.pop()
            if h in seen or h == name:
                continue
            seen.add(h)
            stack.extend(jit[h]["calls"] & jit.keys())
        hc = sum(jit[h]["code"] for h in seen)
        rec = {"lang": "triton", "own": j["code"], "helpers": hc, "total": j["code"] + hc,
               "raw_total": j["raw"] + sum(jit[h]["raw"] for h in seen), "helpers_list": sorted(seen),
               "source_const": "@triton.jit", "note": ""}
        if name in res["kernels"]:
            rec["note"] = "name also a CUDA kernel"
            name = name + "@triton"
        res["kernels"][name] = rec

    # shared helpers: reached by >= 2 kernels
    cnt = {}
    for k in res["kernels"].values():
        for h in k["helpers_list"]:
            cnt[h] = cnt.get(h, 0) + 1
    res["helpers_shared"] = sorted(h for h, v in cnt.items() if v >= 2)
    if not any(k["lang"] == "cuda" for k in res["kernels"].values()):
        res["external"] = True
        res["note"] = "no own CUDA kernels (SGLang/external kernels at this point); own Triton only"
    return res


def main():
    log = git("log", "--format=%h %H %ad %s", "--date=iso-strict", "--", PATH).splitlines()
    out = {}
    for line in reversed(log):
        h, full, date, subj = line.split(" ", 3)
        short = full[:8]
        try:
            r = analyse_commit(full)
        except Exception as e:
            r = {"error": repr(e), "kernels": {}}
        r["date"], r["subject"] = date, subj[:160]
        out[short] = r
        print(short, date[:16], r.get("file", {}).get("lines"), len(r["kernels"]),
              r.get("error", ""), file=sys.stderr)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "loc.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)


if __name__ == "__main__":
    main()
