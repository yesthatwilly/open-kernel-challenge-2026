#!/usr/bin/env python3
"""churn.py — the mechanical half of /churnsubs.

Everything that is not a judgement call lives here: claiming hypotheses, worktrees, lint,
attempt files, recording job ids, harvesting results, classification, attribution, promotion
bookkeeping, the budget rule, indexes. The LLM half (implementing a hypothesis, vetoing a weak
one, reviewing the rules) happens in the skill between these subcommands.

This script NEVER runs `okc check` or `okc submit` itself: it prints the exact command and the
skill runs it as a bare Bash call so the repo's attempt gate fires. All state is in files, git and
okc; nothing is kept in memory between calls.

Exit codes: 0 ok · 1 nothing to do · 2 refused (precondition) · 3 needs a human.
Output: JSON on stdout, log lines on stderr.
"""
import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

# ----------------------------------------------------------------------------- constants
PROBLEMS = ["muse_attention_decode", "muse_bf16_proj", "muse_decode_layer", "muse_moe_w4a16", "muse_prefill_layer"]
ALIASES = {
    "attn": "muse_attention_decode", "attn_decode": "muse_attention_decode", "attention": "muse_attention_decode",
    "bf16": "muse_bf16_proj", "proj": "muse_bf16_proj", "bf16_proj": "muse_bf16_proj",
    "decode": "muse_decode_layer", "decode_layer": "muse_decode_layer",
    "moe": "muse_moe_w4a16", "w4a16": "muse_moe_w4a16",
    "prefill": "muse_prefill_layer", "prefill_layer": "muse_prefill_layer",
}
OUTCOMES = ["gain", "neutral", "regression", "failed", "rejected"]
MAX_IN_FLIGHT = 8          # M: per problem
PER_TICK_CAP = 4           # C: per problem per tick
REWRITE_SHARE = 0.4        # share of each day's checks kept for kind: rewrite (user, 2026-10-24)
GLOBAL_TICK_CAP = 6
SUBMIT_FLOOR = 1.0         # a check-proven gain is promoted with a scored run only if ci.low >= this
STALE_H = 6.0              # queued this long: no longer counts as in flight
LOST_H = 24.0              # queued this long: judge-lost
LOCK_STALE_H = 2.0
ATTEMPT_FRESH_H = 2.0      # the gate's freshness window
AREAS_HINT = ("pre-attention norm+rope", "cache store", "FA3 call", "output norm-gate", "qkv GEMM", "o GEMM",
              "pooler GEMM", "unpooler GEMM", "router GEMM", "routing top-k", "MoE align/dispatch",
              "w13 marlin", "silu-mul", "w2 marlin", "post-expert norm+combine", "attention boundary",
              "FFN boundary", "whole-layer fusion", "launch overhead")

HERE = os.path.dirname(os.path.abspath(__file__))
LINT = os.path.join(HERE, "okc_lint.py")


def log(msg):
    print(msg, file=sys.stderr)


def out(obj, code=0):
    print(json.dumps(obj, indent=1, default=str))
    sys.exit(code)


def refuse(msg, code=2, **extra):
    out({"ok": False, "refused": msg, **extra}, code)


def run(cmd, cwd=None, check=True, inp=None):
    p = subprocess.run(cmd, cwd=cwd, input=inp, text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}): {p.stderr.strip()[:800]}")
    return p.stdout


def git(*args, cwd=None, check=True):
    return run(["git", *args], cwd=cwd or REPO, check=check)


def okc_json(*args):
    txt = run(["okc", *args, "--json"], check=False)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt}


def now():
    return dt.datetime.now(dt.timezone.utc)


def ts(t=None):
    return (t or now()).strftime("%Y-%m-%dT%H%M%SZ")


def hours_to_reset(t=None):
    t = t or now()
    nxt = (t + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - t).total_seconds() / 3600


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def norm_sha256(path):
    """sha256 of the code with docstrings, comments and formatting removed: two files that only
    differ in prose hash the same, which the judge's own dedup does not catch."""
    import ast
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return None
    class Strip(ast.NodeTransformer):
        # drop docstrings, type annotations, TypedDict classes and local-name differences
        def visit_Module(self, n):
            self.generic_visit(n)
            n.body = [b for b in n.body if not (isinstance(b, ast.ClassDef) and any(
                (isinstance(x, ast.Name) and x.id == "TypedDict") for x in b.bases))]
            return self._nodoc(n)

        def _nodoc(self, n):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
            return n

        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            n.returns = None
            for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs:
                a.annotation = None
            if n.args.vararg:
                n.args.vararg.annotation = None
            if n.args.kwarg:
                n.args.kwarg.annotation = None
            return self._nodoc(n)

        def visit_ClassDef(self, n):
            self.generic_visit(n)
            return self._nodoc(n)

        def visit_AnnAssign(self, n):
            if n.value is None:
                return None
            return ast.Assign(targets=[n.target], value=n.value, lineno=0)

        def visit_Name(self, n):
            # local/parameter names are normalised to their binding order would be ideal; as a
            # cheap approximation, names that are plain identifiers of one or two chars or common
            # parameter spellings collapse to a placeholder
            if n.id in ("weights", "state", "data", "cfg", "config", "x", "w", "acc", "layer"):
                return ast.copy_location(ast.Name(id="_", ctx=n.ctx), n)
            return n

    tree = Strip().visit(tree)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def problem_of(name):
    name = name.strip()
    if name in PROBLEMS:
        return name
    if name in ALIASES:
        return ALIASES[name]
    refuse(f"unknown problem {name}; one of {PROBLEMS} or aliases {sorted(ALIASES)}")


# ----------------------------------------------------------------------------- repo paths
REPO = os.environ.get("CHURN_REPO") or run(["git", "rev-parse", "--show-toplevel"], check=False).strip() or os.getcwd()
WORKTREES = os.path.join(REPO, ".worktrees")
LOCK = os.path.join(REPO, ".claude", "churn.lock")
TEMPLATE = os.path.join(REPO, "ATTEMPT_TEMPLATE.md")


def hdir(p, *parts):
    return os.path.join(REPO, p, "hypotheses", *parts)


def rel(path):
    return os.path.relpath(path, REPO)


def list_md(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".md") and f != "README.md") if os.path.isdir(d) else []


def best_path(p):
    return hdir(p, "BEST.json")


def load_best(p):
    f = best_path(p)
    return json.load(open(f)) if os.path.exists(f) else None


def save_best(p, obj):
    json.dump(obj, open(best_path(p), "w"), indent=1)
    open(best_path(p), "a").write("\n")


# ----------------------------------------------------------------------------- frontmatter
def _scalar(v):
    v = v.strip()
    if v in ("", "null", "~", "None"):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if (v[0] == v[-1] and v[0] in "\"'") and len(v) >= 2:
        return v[1:-1]
    if v[0] == "[" and v[-1] == "]":
        inner = v[1:-1].strip()
        return [_scalar(x) for x in inner.split(",")] if inner else []
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d*", v):
        return float(v)
    return v


def parse_fm(text):
    """A small YAML subset: scalars, `key: >`/`|` blocks, lists of scalars or of flat dicts."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    block = text[3:end].strip("\n").splitlines()
    body = text[end + 4:].lstrip("\n")
    fm, i = {}, 0
    while i < len(block):
        line = block[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2)
        if val in (">", "|", ">-", "|-"):
            buf = []
            i += 1
            while i < len(block) and (block[i].startswith("  ") or not block[i].strip()):
                buf.append(block[i].strip())
                i += 1
            fm[key] = (" " if val.startswith(">") else "\n").join(x for x in buf if x).strip()
            continue
        if val == "":
            items, i = [], i + 1
            while i < len(block) and (block[i].startswith("  ") or not block[i].strip()):
                l = block[i]
                if l.strip().startswith("- "):
                    rest = l.strip()[2:]
                    mm = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", rest)
                    if mm:
                        d = {mm.group(1): _scalar(mm.group(2))}
                        i += 1
                        while i < len(block) and block[i].startswith("    ") and not block[i].strip().startswith("- "):
                            m2 = re.match(r"^\s+([A-Za-z_][\w-]*):\s*(.*)$", block[i])
                            if m2:
                                d[m2.group(1)] = _scalar(m2.group(2))
                            i += 1
                        items.append(d)
                        continue
                    items.append(_scalar(rest))
                i += 1
            fm[key] = items
            continue
        fm[key] = _scalar(val)
        i += 1
    return fm, body


def fm_set(path, **updates):
    """Set top-level scalar keys in a file's frontmatter (rewrite or append before the closing ---)."""
    text = open(path).read()
    assert text.startswith("---"), path
    end = text.find("\n---", 3)
    head, tail = text[:end], text[end:]
    lines = head.split("\n")
    for k, v in updates.items():
        if v is None:
            sv = "null"
        elif isinstance(v, bool):
            sv = "true" if v else "false"
        elif isinstance(v, (int, float)):
            sv = repr(v) if isinstance(v, float) else str(v)
        elif isinstance(v, list):
            sv = "[" + ", ".join(str(x) for x in v) + "]"
        else:
            sv = str(v)
            if re.search(r"[:#\[\]{}]|^\s|\s$", sv) or sv == "":
                sv = json.dumps(sv)
        done = False
        for idx, l in enumerate(lines):
            if re.match(rf"^{re.escape(k)}:", l):
                lines[idx] = f"{k}: {sv}"
                done = True
                break
        if not done:
            lines.append(f"{k}: {sv}")
    open(path, "w").write("\n".join(lines) + tail)


def read_hyp(path):
    fm, body = parse_fm(open(path).read())
    fm["_path"] = path
    fm["_body"] = body
    return fm


def set_section(path, heading, content):
    """Replace or append a `## heading` section's body in a markdown file."""
    text = open(path).read()
    pat = re.compile(rf"(^## {re.escape(heading)}\s*\n)(.*?)(?=^## |\Z)", re.S | re.M)
    new = f"## {heading}\n\n{content.rstrip()}\n\n"
    if pat.search(text):
        text = pat.sub(lambda m: new, text, count=1)
    else:
        text = text.rstrip("\n") + "\n\n" + new
    open(path, "w").write(text)


def get_section(text, heading):
    m = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


# ----------------------------------------------------------------------------- hypothesis files
REQUIRED = ["problem", "slug", "created", "area", "change", "priority", "status"]


def find_hyp(p, slug):
    for sub in ["queue", "running"] + [f"done/{o}" for o in OUTCOMES]:
        for f in list_md(hdir(p, sub)):
            if f.endswith(f"-{slug}.md") or re.sub(r"^\d{4}-\d{2}-\d{2}T\d{4,6}Z-", "", f)[:-3] == slug:
                return os.path.join(hdir(p, sub), f), sub
    return None, None


def all_hyps(p):
    res = []
    for sub in ["queue", "running"] + [f"done/{o}" for o in OUTCOMES]:
        for f in list_md(hdir(p, sub)):
            h = read_hyp(os.path.join(hdir(p, sub), f))
            h["_where"] = sub
            res.append(h)
    return res


def lint_hyp(path, p=None):
    errs = []
    h = read_hyp(path)
    for k in REQUIRED:
        if h.get(k) in (None, ""):
            errs.append(f"missing {k}")
    if h.get("problem") and h["problem"] not in PROBLEMS:
        errs.append(f"problem {h['problem']} unknown")
    if p and h.get("problem") != p:
        errs.append(f"problem {h.get('problem')} is not {p}")
    slug = str(h.get("slug") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,60}", slug):
        errs.append("slug must match [a-z0-9-], 3..61 chars")
    if not os.path.basename(path).endswith(f"-{slug}.md"):
        errs.append(f"filename must end with -{slug}.md")
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{4,6}Z-", os.path.basename(path)):
        errs.append("filename must start with <YYYY-MM-DDTHHMMSSZ>-")
    if h.get("priority") not in (1, 2, 3, 4, 5):
        errs.append("priority must be 1..5")
    if h.get("status") not in ("queued", "hold", "running", "done"):
        errs.append("status must be queued|hold|running|done")
    if h.get("risk") not in (None, "low", "medium", "high"):
        errs.append("risk must be low|medium|high")
    if len(str(h.get("change") or "")) < 40:
        errs.append("change must be a real paragraph (>= 40 chars)")
    exp = h.get("expected")
    if exp is not None and not isinstance(exp, list):
        errs.append("expected must be a list of {shapes, direction, why}")
    if h.get("depends_on"):
        dp, _ = find_hyp(h["problem"], str(h["depends_on"])) if h.get("problem") in PROBLEMS else (None, None)
        if not dp:
            errs.append(f"depends_on {h['depends_on']} not found")
    # slug uniqueness across the problem
    if h.get("problem") in PROBLEMS and slug:
        hits = [x for x in all_hyps(h["problem"]) if x.get("slug") == slug and os.path.abspath(x["_path"]) != os.path.abspath(path)]
        if hits:
            errs.append(f"slug {slug} already used by {rel(hits[0]['_path'])}")
    return errs, h


# ----------------------------------------------------------------------------- okc helpers
def quotas():
    d = okc_json("problems")
    probs = d.get("problems", d) if isinstance(d, dict) else d
    q = {}
    for x in probs if isinstance(probs, list) else []:
        q[x.get("name")] = x.get("limits") or {}
    return q, (d.get("tracks") if isinstance(d, dict) else None)


def jobs_list():
    d = okc_json("jobs")
    return d.get("jobs", d) if isinstance(d, dict) else d


def local_jobs():
    f = os.path.expanduser("~/.config/okc/jobs.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def job_status(job):
    return okc_json("status", job)


# ----------------------------------------------------------------------------- lock
def take_lock():
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    if os.path.exists(LOCK):
        try:
            info = json.load(open(LOCK))
        except Exception:
            info = {}
        age_h = (time.time() - info.get("at", 0)) / 3600
        alive = info.get("pid") and os.path.exists(f"/proc/{info['pid']}")
        if age_h < LOCK_STALE_H and alive:
            refuse(f"a churn tick is running since {info.get('started')} (pid {info.get('pid')}); lock {rel(LOCK)}")
        log(f"taking over stale lock from {info.get('started')}")
    json.dump({"pid": os.getppid(), "at": time.time(), "started": ts()}, open(LOCK, "w"))


def release_lock():
    if os.path.exists(LOCK):
        os.remove(LOCK)


# ----------------------------------------------------------------------------- attempts
def write_attempt(tree, p, slug, mode, h, impl_notes, rules_check, risk_text, goal):
    """Write a fresh attempt file in <tree>/<p>/attempts from the template; returns its relative name."""
    tpl = open(TEMPLATE).read()
    name = f"{ts()}-{mode}-{slug}.md"
    path = os.path.join(tree, p, "attempts", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head_sha = git("rev-parse", "--short", "HEAD", cwd=tree).strip()
    text = f"# {p} · {name[:-3].split('-' + mode + '-')[0]} · {mode} · {slug}\n\n"
    text += f"Hypothesis: `{rel(h['_path'])}` (branch `{h.get('branch') or 'master'}`).\n\n"
    text += f"- **Job:** `<job id>` (`okc status <job>`)\n- **File:** `{p}/submission.py` at git `{head_sha}`\n\n"
    text += f"## Goal\n{goal.strip()}\n\n## What changed, and where\n{impl_notes.strip() or '(see hypothesis)'}\n\n"
    rc = rules_check.strip()
    if not rc:
        rc = get_section(tpl, "Rules check (each must be true before the run)")
    text += f"## Rules check (each must be true before the run)\n{rc}\n\n"
    text += f"## Risks / what could fail\n{risk_text.strip() or '(see hypothesis)'}\n\n## Result (after the run)\n(pending)\n"
    open(path, "w").write(text)
    return name


def fill_attempt_job(path, job):
    s = open(path).read().replace("`<job id>`", f"`{job}`", 1)
    open(path, "w").write(s)


# ----------------------------------------------------------------------------- results
def shape_key(case):
    """`k=v` pairs, space-joined; spaces inside values become `_` so keys split cleanly."""
    return " ".join(f"{k}={str(v).replace(' ', '_')}" for k, v in sorted(case.items()) if k not in ("seed", "weight"))


def trim_result(job, status_json, p, h):
    r = status_json.get("result") or {}
    recs = r.get("records") or []
    tests = [x for x in recs if x.get("event") == "test"]
    bench = [x for x in recs if x.get("event") == "benchmark"]
    profs = {x.get("index"): x for x in recs if x.get("event") == "profile"}
    first_fail = next(({"index": x.get("index"), "case": x.get("case"), "message": x.get("message")}
                       for x in tests if x.get("status") != "pass"), None)
    if first_fail is None and r.get("error"):
        first_fail = {"index": None, "case": None, "message": str(r.get("error"))[:400]}
    shapes = []
    for b in bench:
        top = []
        rep = (profs.get(b.get("index")) or {}).get("report") or ""
        for line in rep.splitlines()[1:]:
            m = re.match(r"^(.*?)\s{2,}([\d.]+) µs\s+([\d.]+)%\s+(\d+)\s*$", line)
            if m and m.group(1).strip() != "total":
                top.append([m.group(1).strip()[:60], float(m.group(2)), float(m.group(3))])
            if len(top) >= 4:
                break
        shapes.append({
            "key": shape_key(b.get("case") or {}), "load": (b.get("case") or {}).get("load"),
            "case": b.get("case"), "weight": (b.get("case") or {}).get("weight"),
            "median_ns": b.get("median_ns"), "baseline_ns": b.get("baseline_ns"),
            "speedup": b.get("speedup"), "speedup_of_medians": b.get("speedup_of_medians"),
            "err_ns": b.get("err_ns"), "speed_of_light_ns": b.get("speed_of_light_ns"),
            "status": b.get("status"), "top_kernels": top,
        })
    loads = {}
    for s in shapes:
        w = s.get("weight") or 1
        d = loads.setdefault(s.get("load") or "?", {"load": s.get("load"), "sglang_ns": 0.0, "yours_ns": 0.0})
        d["sglang_ns"] += w * (s.get("baseline_ns") or 0)
        d["yours_ns"] += w * (s.get("median_ns") or 0)
    for d in loads.values():
        d["speedup"] = (d["sglang_ns"] / d["yours_ns"]) if d["yours_ns"] else None
    return {
        "schema": 1, "problem": p, "slug": h.get("slug"), "job": job, "mode": h.get("mode") or status_json.get("mode"),
        "duplicate_of": status_json.get("duplicate_of"), "content_hash": status_json.get("content_hash"),
        "submission_sha256": h.get("submission_sha256"), "submission_blob": h.get("submission_blob"),
        "impl_commit": h.get("impl_commit"), "base_sha": h.get("base_sha"),
        "created_at": status_json.get("created_at"), "finished_at": status_json.get("finished_at"),
        "gpu": r.get("gpu"), "evaluator_sha256": (r.get("evaluator") or {}).get("code_sha256"),
        "status": status_json.get("status"), "passed": r.get("passed"), "error": r.get("error"),
        "returncode": r.get("returncode"),
        "tests": {"total": len(tests), "passed": sum(1 for x in tests if x.get("status") == "pass")},
        "first_failing": first_fail,
        "speedup": r.get("speedup"), "speedup_ci": r.get("speedup_ci"), "score_us": r.get("score_us"),
        "baseline_us": r.get("baseline_us"), "flags": r.get("flags") or [], "checks": r.get("checks") or {},
        "shapes": shapes, "loads": sorted(loads.values(), key=lambda d: str(d["load"])),
        "attribution": None,
    }


def base_result_for(p, h, best):
    """The result the hypothesis should be compared against: same submission blob as its base."""
    blob = h.get("base_blob")
    if best and blob and best.get("submission_blob") == blob and best.get("result"):
        f = os.path.join(REPO, best["result"])
        if os.path.exists(f):
            return json.load(open(f)), best["result"]
    for o in OUTCOMES:
        d = hdir(p, "done", o)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".result.json"):
                try:
                    rj = json.load(open(os.path.join(d, f)))
                except Exception:
                    continue
                if blob and rj.get("submission_blob") == blob and rj.get("passed"):
                    return rj, rel(os.path.join(d, f))
    return None, None


def attribute(res, base, h):
    s, ci = res.get("speedup") or 0, res.get("speedup_ci") or [0, 0]
    noise = max(0.015, (ci[1] - ci[0]) / (2 * s)) if s else 0.015
    bmap = {x["key"]: x for x in (base or {}).get("shapes", [])}
    rows, gained, lost = [], [], []
    for x in res["shapes"]:
        bx = bmap.get(x["key"])
        b_speed = (bx or {}).get("speedup") or 1.0
        b_us = ((bx or {}).get("median_ns") or x.get("baseline_ns") or 0) / 1000
        delta = (x.get("speedup") or 0) / b_speed if b_speed else None
        v = "flat"
        if delta is not None:
            if delta > 1 + 2 * noise:
                v, _ = "gained", gained.append(x["key"])
            elif delta < 1 - 2 * noise:
                v, _ = "lost", lost.append(x["key"])
        rows.append({"key": x["key"], "weight": x.get("weight"), "base_us": round(b_us, 2),
                     "hyp_us": round((x.get("median_ns") or 0) / 1000, 2), "delta": round(delta, 4) if delta else None, "verdict": v})

    def grp(keys):
        g = {}
        for k in keys:
            kv = dict(t.split("=", 1) for t in k.split(" ") if "=" in t)
            g.setdefault((kv.get("layer") or kv.get("load") or "?"), []).append(kv.get("load") or kv.get("m") or "")
        return "; ".join(f"{layer} {', '.join(sorted(set(v)))}" for layer, v in g.items())

    summary = []
    if gained:
        summary.append("gained on " + grp(gained))
    if lost:
        summary.append("lost on " + grp(lost))
    if not summary:
        summary.append("no shape moved outside noise")
    hit = None
    exp = h.get("expected") or []
    if isinstance(exp, list) and exp:
        ok = 0
        for e in exp:
            if not isinstance(e, dict):
                continue
            pat, want = str(e.get("shapes") or "*"), str(e.get("direction") or "faster")
            matched = [r for r in rows if fnmatch.fnmatch(r["key"], pat)]
            if matched:
                vs = {r["verdict"] for r in matched}
                good = {"faster": "gained", "slower": "lost", "flat": "flat"}.get(want, "gained")
                if good in vs and ("lost" if good == "gained" else "gained") not in vs:
                    ok += 1
        hit = f"{ok}/{len([e for e in exp if isinstance(e, dict)])}"
    return {"base_result": (base or {}).get("_path"), "noise": round(noise, 4), "shapes": rows,
            "summary": "; ".join(summary), "expected_hit": hit}


def classify(res, best, base):
    s, ci = res.get("speedup"), res.get("speedup_ci") or [None, None]
    lo, hi = ci[0], ci[1]
    checks = res.get("checks") or {}
    if res.get("status") == "rejected" or checks.get("rejected"):
        return "rejected", "judge:" + (str((checks.get("rejected") or ["rejected"])[0])[:80])
    if res.get("status") == "failed" or not res.get("passed") or res.get("error"):
        ff = res.get("first_failing") or {}
        return "failed", "test:" + (json.dumps(ff.get("case")) if ff.get("case") else str(ff.get("message") or res.get("error") or "failed")[:80])
    if s is None or lo is None:
        return "failed", "no-speedup-in-result"
    B = (best or {}).get("speedup") or 0.0
    Bhi = ((best or {}).get("ci") or [B, B])[1] or B
    b = (base or {}).get("speedup") if base else None
    if b is None:
        b = B
    elif b > B:
        # A base merged without a scored submit can be ahead of BEST.json; a gain must beat it too.
        B, Bhi = b, ((base or {}).get("speedup_ci") or [b, b])[1] or b
    if lo > max(B, Bhi) and s > B:
        return "gain", "gain"
    if s > B:
        return "neutral", "unproven-vs-best"
    if hi < b:
        return "regression", "regression"
    if lo > b:
        return "neutral", "beats-base-not-best"
    return "neutral", "within-ci"


# ----------------------------------------------------------------------------- subcommands
def cmd_preflight(a):
    take_lock()
    br = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if br != "master":
        release_lock()
        refuse(f"main tree is on {br}, not master")
    who = run(["okc", "whoami"], check=False).strip()
    if not who or "not signed in" in who.lower():
        release_lock()
        refuse("okc is not signed in")
    q, tracks = quotas()
    out({"ok": True, "branch": br, "whoami": who.splitlines()[0][:80], "quotas": q, "tracks": tracks,
         "hours_to_reset": round(hours_to_reset(), 2), "lock": rel(LOCK)})


def cmd_unlock(a):
    release_lock()
    out({"ok": True})


def cmd_intake(a):
    added, skipped = [], []
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *[f"{p}/hypotheses/queue" for p in PROBLEMS]).split()
    for f in untracked:
        if not f.endswith(".md") or os.path.basename(f) == "README.md":
            continue
        path = os.path.join(REPO, f)
        errs, h = lint_hyp(path)
        if errs:
            skipped.append({"file": f, "errors": errs})
            continue
        git("add", "--", f)
        added.append(f)
    if added:
        git("commit", "-q", "-m", f"churn: intake {len(added)} hypotheses\n\n" + "\n".join(added))
    out({"ok": True, "added": added, "skipped": skipped}, 0 if (added or not skipped) else 0)


def cmd_lint_hyp(a):
    errs, h = lint_hyp(a.file, problem_of(a.problem) if a.problem else None)
    out({"ok": not errs, "file": a.file, "errors": errs, "slug": h.get("slug"), "problem": h.get("problem")}, 0 if not errs else 1)


def running_hyps(p):
    return [read_hyp(os.path.join(hdir(p, "running"), f)) for f in list_md(hdir(p, "running"))]


def cmd_recover(a):
    actions, needs_human = [], []
    git("worktree", "prune")
    wts = git("worktree", "list", "--porcelain")
    wt_paths = set(re.findall(r"^worktree (.+)$", wts, re.M))
    branches = set(git("branch", "--list", "hyp/*", "--format=%(refname:short)").split())
    lj = local_jobs()
    remote = jobs_list()
    for p in PROBLEMS:
        for h in running_hyps(p):
            slug, path = h.get("slug"), h["_path"]
            br = h.get("branch") or f"hyp/{p}/{slug}"
            wt = os.path.join(WORKTREES, p, slug)
            if h.get("stage") in ("claimed", "implemented") or (h.get("stage") == "dispatched" and not h.get("job")):
                if not os.path.isdir(wt) and br in branches:
                    git("worktree", "add", wt, br)
                    actions.append(f"{p}/{slug}: recreated worktree from {br}")
                elif not os.path.isdir(wt) and h.get("base_sha"):
                    if h.get("stage") == "claimed":
                        git("worktree", "add", "-b", br, wt, h["base_sha"])
                        actions.append(f"{p}/{slug}: recreated worktree+branch at base")
                    else:
                        needs_human.append(f"{p}/{slug}: stage {h.get('stage')} but branch and worktree are gone")
                        fm_set(path, needs_human=True)
                        continue
                elif os.path.isdir(wt) and br not in branches:
                    git("checkout", "-q", "-b", br, cwd=wt)
                    actions.append(f"{p}/{slug}: recreated branch in worktree")
            if not h.get("job") and h.get("stage") in ("implemented", "dispatched"):
                # orphan adoption: a run was started but its id never recorded
                sub = os.path.join(wt, p, "submission.py")
                cands = [(j, v) for j, v in lj.items() if v.get("file") == sub and v.get("at", 0) >= _epoch(h.get("claimed_at"))]
                if len(cands) == 1:
                    job = cands[0][0]
                    _record(p, h, job, note="adopted from ~/.config/okc/jobs.json")
                    actions.append(f"{p}/{slug}: adopted orphan job {job}")
                    continue
                # fallback: unreferenced remote jobs for this problem in the window
                known = _known_jobs()
                orphans = [j for j in remote if j.get("problem") == p and j.get("job_id") not in known
                           and j.get("submitted_at", 0) >= _epoch(h.get("implemented_at") or h.get("claimed_at"))]
                unid = [x for x in running_hyps(p) if not x.get("job") and x.get("stage") in ("implemented", "dispatched")]
                if len(orphans) == 1 and len(unid) == 1:
                    _record(p, h, orphans[0]["job_id"], note="adopted from okc jobs (single orphan)")
                    actions.append(f"{p}/{slug}: adopted orphan job {orphans[0]['job_id']} by time window")
                elif orphans:
                    needs_human.append(f"{p}/{slug}: {len(orphans)} unreferenced okc jobs vs {len(unid)} unidentified hypotheses")
                    fm_set(path, needs_human=True)
    out({"ok": not needs_human, "actions": actions, "needs_human": needs_human}, 3 if needs_human else 0)


def _epoch(s):
    if not s:
        return 0
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except ValueError:
        return 0


def _known_jobs():
    known = set()
    for p in PROBLEMS:
        for h in all_hyps(p):
            for k in ("job", "promotion_job"):
                if h.get(k):
                    known.add(str(h[k]))
        ad = os.path.join(REPO, p, "attempts")
        for f in os.listdir(ad) if os.path.isdir(ad) else []:
            if f.endswith(".md"):
                for m in re.finditer(r"\*\*Job:\*\*\s*`([0-9a-f]{16})`", open(os.path.join(ad, f)).read()):
                    known.add(m.group(1))
    txt = git("grep", "-h", "-o", r"\*\*Job:\*\* `[0-9a-f]\{16\}`", "--", "*/attempts/*.md", check=False)
    known |= set(re.findall(r"`([0-9a-f]{16})`", txt))
    for br in git("branch", "--list", "hyp/*", "--format=%(refname:short)").split():
        txt = git("grep", "-h", "-o", r"\*\*Job:\*\* `[0-9a-f]\{16\}`", br, "--", "*/attempts/*.md", check=False)
        known |= set(re.findall(r"`([0-9a-f]{16})`", txt))
    return known


def _record(p, h, job, note=None):
    """Write the job id into the attempt file (branch) and the running md (master); commit both."""
    slug = h["slug"]
    wt = os.path.join(WORKTREES, p, slug)
    af = h.get("attempt_file")
    if af and os.path.isdir(wt):
        apath = os.path.join(wt, p, "attempts", af)
        if os.path.exists(apath):
            fill_attempt_job(apath, job)
            git("add", "--", os.path.join(p, "attempts", af), cwd=wt)
            git("commit", "-q", "-m", f"attempt: {slug} job {job}", cwd=wt, check=False)
    fm_set(h["_path"], job=job, stage="dispatched", dispatched_at=h.get("dispatched_at") or ts(), status="running")
    if note:
        fm_set(h["_path"], recover_note=note)
    git("add", "--", rel(h["_path"]))
    git("commit", "-q", "-m", f"churn({p}): dispatched {slug} as {job}", check=False)


def _terminal(st):
    return st in ("done", "failed", "rejected", "duplicate")


def cmd_harvest(a):
    probs = [problem_of(a.problem)] if a.problem else PROBLEMS
    harvested, pending = [], []
    for p in probs:
        for h in running_hyps(p):
            if h.get("stage") != "dispatched" or not h.get("job"):
                continue
            job = str(h["job"])
            sj = job_status(job)
            st = sj.get("status")
            age_h = (time.time() - _epoch(h.get("dispatched_at"))) / 3600
            if not _terminal(st):
                if age_h > LOST_H:
                    sj = {"status": "failed", "result": {"passed": False, "error": "judge-lost: queued > 24 h"}, "mode": h.get("mode")}
                    st = "failed"
                else:
                    if age_h > STALE_H and not h.get("stale"):
                        fm_set(h["_path"], stale=True)
                    pending.append({"problem": p, "slug": h["slug"], "job": job, "status": st, "age_h": round(age_h, 1)})
                    continue
            if st == "duplicate" and sj.get("duplicate_of"):
                orig = job_status(str(sj["duplicate_of"]))
                if _terminal(orig.get("status")) and orig.get("status") != "duplicate":
                    orig["duplicate_of"] = sj["duplicate_of"]
                    sj = orig
                else:
                    pending.append({"problem": p, "slug": h["slug"], "job": job, "status": "duplicate-pending"})
                    continue
            harvested.append(_harvest_one(p, h, job, sj))
    out({"ok": True, "harvested": harvested, "pending": pending}, 0 if (harvested or pending) else 1)


def archive_job(p, slug, job, sj):
    """Keep everything okc returned for a run: the raw status JSON (all records) and the profile
    files okc writes next to the submission, which vanish with the worktree. Lives at
    <p>/hypotheses/archive/<slug>/ and is committed with the harvest."""
    import gzip
    adir = hdir(p, "archive", slug)
    os.makedirs(adir, exist_ok=True)
    with gzip.open(os.path.join(adir, f"{job}.okc.json.gz"), "wt") as f:
        json.dump(sj, f)
    paths = dict(sj.get("profile") or {})
    paths.update({f"log": (sj.get("output") or {}).get("log")})
    for kind, src in paths.items():
        if not src or not os.path.exists(src) or os.path.getsize(src) == 0:
            continue
        dst = os.path.join(adir, {"text": f"{job}.profile.txt", "log": f"{job}.log"}.get(kind, f"{job}.profile.{kind}"))
        if kind == "json":
            with open(src, "rb") as fi, gzip.open(dst + ".gz", "wb") as fo:
                fo.write(fi.read())
        else:
            shutil.copyfile(src, dst)
    return adir


def _harvest_one(p, h, job, sj):
    slug = h["slug"]
    best = load_best(p)
    res = trim_result(job, sj, p, h)
    base, base_path = base_result_for(p, h, best)
    if base:
        base["_path"] = base_path
    res["attribution"] = attribute(res, base, h) if res.get("shapes") else None
    outcome, detail = classify(res, best, base)
    if res.get("mode") == "submit" and outcome == "gain":
        pass  # promotion bookkeeping below
    ddir = hdir(p, "done", outcome)
    os.makedirs(ddir, exist_ok=True)
    new_md = os.path.join(ddir, os.path.basename(h["_path"]))
    git("mv", rel(h["_path"]), rel(new_md))
    # patch + result json
    patch = ""
    if h.get("base_sha") and h.get("impl_commit"):
        patch = git("diff", h["base_sha"], h["impl_commit"], "--", f"{p}/submission.py", check=False)
    open(os.path.join(ddir, f"{slug}.patch"), "w").write(patch)
    rj = os.path.join(ddir, f"{slug}.result.json")
    json.dump(res, open(rj, "w"), indent=1)
    open(rj, "a").write("\n")
    # sections
    s, ci = res.get("speedup"), res.get("speedup_ci") or [None, None]
    lines = [f"- job `{job}` ({res.get('mode')}), status {res.get('status')}, passed {res.get('passed')}, tests {res['tests']['passed']}/{res['tests']['total']}"]
    if res.get("duplicate_of"):
        lines.append(f"- duplicate of `{res['duplicate_of']}` (same code already judged)")
    if res.get("first_failing"):
        lines.append(f"- first failing: {json.dumps(res['first_failing'].get('case'))}: {res['first_failing'].get('message')}")
    if s is not None:
        lines.append(f"- speedup {s:.4f}x, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}], score {res.get('score_us')} µs vs SGLang {res.get('baseline_us')} µs")
    lines.append(f"- best at harvest: {json.dumps({k: best.get(k) for k in ('speedup', 'ci', 'job')} ) if best else 'none'}; base result: {base_path or 'SGLang (no base result)'}")
    lines.append(f"- **outcome: {outcome}** ({detail})")
    lines.append(f"- files: `{rel(rj)}`, `{rel(os.path.join(ddir, slug + '.patch'))}`; branch `{h.get('branch')}`")
    set_section(new_md, "Result", "\n".join(lines))
    if res.get("attribution"):
        at = res["attribution"]
        tbl = ["| shape | w | base µs | hyp µs | delta | verdict |", "|---|---|---|---|---|---|"]
        for r in at["shapes"]:
            tbl.append(f"| {r['key']} | {r['weight']} | {r['base_us']} | {r['hyp_us']} | {r['delta']} | {r['verdict']} |")
        extra = f"\n\n{at['summary']}." + (f" Expected hit: {at['expected_hit']}." if at.get("expected_hit") else "") + f" Noise ±{at['noise']*100:.1f}%."
        set_section(new_md, "Attribution", "\n".join(tbl) + extra)
    fu = []
    if str(h.get("kind") or "") == "rewrite":
        fu.append(f"- rewrite stage {h.get('rewrite_stage') or 1} of `{h.get('replaces') or h.get('area')}`: read the replaced kernels' own us per shape "
                  f"(profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is "
                  f"`/hypothesize {p} \"<what to fix or tune>\" --kind rewrite --depends-on {slug} --stage {int(h.get('rewrite_stage') or 1) + 1}`; "
                  "a slow or failed first cut is the expected start, not a do-not-retry.")
    if outcome == "failed" and res.get("first_failing"):
        fu.append(f"- first failing case {json.dumps(res['first_failing'].get('case'))}: {res['first_failing'].get('message')}. Suggest `/hypothesize {p} \"fix {slug}: ...\"` with the fix as its own change.")
    elif outcome == "neutral":
        fu.append(f"- {detail}: the change is not proven either way; a stacked hypothesis (`depends_on: {slug}`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).")
    elif outcome == "regression" and str(h.get("kind") or "") != "rewrite":
        fu.append("- record which shapes lost (Attribution) and do not retry this area in this form.")
    elif outcome == "gain":
        fu.append("- candidate for promotion; stack further changes with `depends_on` once it lands.")
    set_section(new_md, "Follow-ups", "\n".join(fu) or "- none")
    fm_set(new_md, stage="harvested", status="done", outcome=outcome, verdict_detail=detail,
           speedup=s, speedup_ci=ci if s is not None else None, harvested_at=ts(),
           best_at_harvest=(best or {}).get("speedup"),
           promotion=("pending" if outcome == "gain" and res.get("mode") != "submit" else
                      ("onboard" if outcome == "gain" else None)))
    # carry the attempt file from the branch
    if h.get("attempt_file") and h.get("branch"):
        git("checkout", h["branch"], "--", f"{p}/attempts/{h['attempt_file']}", check=False)
        ap = os.path.join(REPO, p, "attempts", h["attempt_file"])
        if os.path.exists(ap):
            txt = open(ap).read()
            txt = re.sub(r"## Result \(after the run\)\n.*\Z",
                         "## Result (after the run)\n" + "\n".join(lines) + "\n", txt, flags=re.S)
            open(ap, "w").write(txt)
            git("add", "--", f"{p}/attempts/{h['attempt_file']}")
    try:
        git("add", "-A", "--", rel(archive_job(p, slug, job, sj)))
    except Exception as e:  # archiving must never block a harvest
        print(f"warning: archive of {slug} failed: {e}", file=sys.stderr)
    git("add", "-A", "--", rel(ddir), rel(hdir(p, "running")))
    msg = f"churn({p}): {slug} → {outcome} (job {job}" + (f", {s:.4f}x [{ci[0]:.4f},{ci[1]:.4f}]" if s is not None else "") + ")"
    git("commit", "-q", "-m", msg)
    # a gain measured by a scored run is already on the board: land it on master now
    if outcome == "gain" and res.get("mode") == "submit":
        _land(p, read_hyp(new_md), res, rel(rj), submitted_job=job)
    # worktree cleanup (branch stays)
    wt = os.path.join(WORKTREES, p, slug)
    keep = outcome == "gain"
    if os.path.isdir(wt) and not keep:
        git("worktree", "remove", "--force", wt, check=False)
    return {"problem": p, "slug": slug, "job": job, "outcome": outcome, "detail": detail, "speedup": s, "ci": ci, "md": rel(new_md)}


def _master_blob(p):
    return git("rev-parse", f"HEAD:{p}/submission.py").strip()


def _dirty(p):
    return bool(git("status", "--porcelain", "--", f"{p}/submission.py").strip())


def _land(p, h, res, result_rel, submitted_job=None):
    """Put a gain's submission.py on master (fast path only) and update BEST.json."""
    slug = h["slug"]
    if _dirty(p):
        fm_set(h["_path"], promotion="pending", promotion_reason="dirty-master")
        git("add", "--", rel(h["_path"]))
        git("commit", "-q", "-m", f"churn({p}): {slug} promotion deferred (dirty master)", check=False)
        return False
    if _master_blob(p) != h.get("base_blob"):
        fm_set(h["_path"], promotion="unmerged", promotion_reason="master moved since base; merge by hand")
        git("add", "--", rel(h["_path"]))
        git("commit", "-q", "-m", f"churn({p}): {slug} gain left unmerged (master moved)", check=False)
        return False
    git("checkout", h["impl_commit"], "--", f"{p}/submission.py")
    lint = subprocess.run([sys.executable, "-I", LINT, os.path.join(REPO, p, "submission.py"), "--problem", p, "--json"], capture_output=True, text=True)
    if lint.returncode != 0:
        git("checkout", "HEAD", "--", f"{p}/submission.py")
        fm_set(h["_path"], promotion="unmerged", promotion_reason="lint-after-merge")
        git("add", "--", rel(h["_path"]))
        git("commit", "-q", "-m", f"churn({p}): {slug} unmerged (lint)", check=False)
        return False
    s, ci = res.get("speedup"), res.get("speedup_ci")
    git("add", "--", f"{p}/submission.py")
    git("commit", "-q", "-m", f"promote({p}): {slug} {s:.4f}x [{ci[0]:.4f},{ci[1]:.4f}] job {res.get('job')}")
    commit = git("rev-parse", "HEAD").strip()
    if submitted_job:
        save_best(p, {"problem": p, "job": submitted_job, "check_job": None, "mode": "submit", "speedup": s, "ci": ci,
                      "score_us": res.get("score_us"), "commit": commit, "submission_blob": _master_blob(p),
                      "slug": slug, "result": result_rel, "updated_at": ts()})
        fm_set(h["_path"], promotion="done", promotion_commit=commit, promotion_job=submitted_job)
    else:
        save_best(p, {"problem": p, "job": res.get("job"), "check_job": res.get("job"), "mode": "check", "speedup": s,
                      "ci": ci, "score_us": res.get("score_us"), "commit": commit, "submission_blob": _master_blob(p),
                      "slug": slug, "result": result_rel, "updated_at": ts()})
        fm_set(h["_path"], promotion="merged", promotion_commit=commit)
    git("add", "--", rel(h["_path"]), rel(best_path(p)))
    git("commit", "-q", "-m", f"churn({p}): {slug} landed on master", check=False)
    return True


def cmd_promote(a):
    """Merge check-proven gains into master and print the scored submit to run (one per problem)."""
    probs = [problem_of(a.problem)] if a.problem else PROBLEMS
    q, _ = quotas()
    plan, skipped = [], []
    for p in probs:
        best = load_best(p)
        cands = []
        for o in ["gain"]:
            for f in list_md(hdir(p, "done", o)):
                h = read_hyp(os.path.join(hdir(p, "done", o), f))
                if h.get("promotion") in (None, "pending", "merged") and h.get("mode") != "submit":
                    cands.append(h)
        if not cands:
            continue
        if any(read_hyp(os.path.join(hdir(p, "done", "gain"), f)).get("promotion") == "submitted" for f in list_md(hdir(p, "done", "gain"))):
            skipped.append({"problem": p, "reason": "a promotion is still unharvested"})
            continue
        cands.sort(key=lambda h: -(h.get("speedup") or 0))
        h = cands[0]
        s, ci = h.get("speedup") or 0, h.get("speedup_ci") or [0, 0]
        B = (best or {}).get("speedup") or 0
        Bhi = ((best or {}).get("ci") or [B, B])[1] or B
        if ci[0] <= max(B, Bhi) and (best or {}).get("slug") != h["slug"]:
            fm_set(h["_path"], promotion="superseded", promotion_reason=f"best is now {B}")
            git("add", "--", rel(h["_path"]))
            git("commit", "-q", "-m", f"churn({p}): {h['slug']} superseded", check=False)
            skipped.append({"problem": p, "slug": h["slug"], "reason": "superseded by current best"})
            continue
        rj = os.path.join(hdir(p, "done", "gain"), f"{h['slug']}.result.json")
        res = json.load(open(rj)) if os.path.exists(rj) else {"speedup": s, "speedup_ci": ci, "job": h.get("job")}
        if h.get("promotion") != "merged":
            if not _land(p, h, res, rel(rj)):
                skipped.append({"problem": p, "slug": h["slug"], "reason": read_hyp(h["_path"]).get("promotion_reason")})
                continue
            h = read_hyp(h["_path"])
        if ci[0] < SUBMIT_FLOOR:
            fm_set(h["_path"], promotion="merged-unsubmitted", promotion_reason=f"ci.low {ci[0]:.3f} < submit floor {SUBMIT_FLOOR}")
            git("add", "--", rel(h["_path"]))
            git("commit", "-q", "-m", f"churn({p}): {h['slug']} merged, not submitted (below floor)", check=False)
            skipped.append({"problem": p, "slug": h["slug"], "reason": "below submit floor; merged only"})
            continue
        left = ((q.get(p) or {}).get("scored") or {}).get("left", 0)
        if left <= 0:
            skipped.append({"problem": p, "slug": h["slug"], "reason": "no scored runs left today; merged, submit tomorrow"})
            continue
        # submit-mode attempt file in the main tree
        impl = get_section(h["_body"], "Implementation notes")
        rules = get_section(h["_body"], "Rules check")
        name = write_attempt(REPO, p, h["slug"], "submit", h, impl, rules, str(h.get("risk") or ""),
                             f"Promote check-proven gain {h['slug']} ({s:.4f}x [{ci[0]:.4f},{ci[1]:.4f}], check job {h.get('job')}) to the board.\n\n{h.get('change')}")
        fm_set(h["_path"], promotion_attempt=name)
        git("add", "--", f"{p}/attempts/{name}", rel(h["_path"]))
        git("commit", "-q", "-m", f"churn({p}): submit attempt for {h['slug']}")
        plan.append({"problem": p, "slug": h["slug"],
                     "run": f"okc submit {p} {os.path.join(REPO, p, 'submission.py')} --no-wait --json",
                     "then": f"churn.py record-submit {p} {h['slug']} <job>"})
    out({"ok": True, "promote": plan, "skipped": skipped}, 0 if plan else 1)


def cmd_record_submit(a):
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path:
        refuse(f"no hypothesis {a.slug} in {p}")
    h = read_hyp(path)
    job = _job_from(a.job)
    name = h.get("promotion_attempt")
    if name:
        fill_attempt_job(os.path.join(REPO, p, "attempts", name), job)
        git("add", "--", f"{p}/attempts/{name}")
    fm_set(path, promotion="submitted", promotion_job=job, promotion_at=ts())
    git("add", "--", rel(path))
    git("commit", "-q", "-m", f"churn({p}): {a.slug} submitted as {job}")
    out({"ok": True, "problem": p, "slug": a.slug, "job": job})


def _job_from(s):
    s = s.strip()
    if re.fullmatch(r"[0-9a-f]{16}", s):
        return s
    try:
        d = json.loads(s)
        return d.get("job_id") or d.get("job") or d.get("id")
    except json.JSONDecodeError:
        m = re.search(r"\b[0-9a-f]{16}\b", s)
        return m.group(0) if m else refuse(f"no job id in {s[:80]}")


def cmd_harvest_promotions(a):
    done = []
    for p in PROBLEMS:
        # neutral too: a gain merged by hand from a neutral result is promoted from done/neutral
        for d, f in [(d, f) for d in ("gain", "neutral") for f in list_md(hdir(p, "done", d))]:
            path = os.path.join(hdir(p, "done", d), f)
            h = read_hyp(path)
            if h.get("promotion") != "submitted" or not h.get("promotion_job"):
                continue
            job = str(h["promotion_job"])
            sj = job_status(job)
            st = sj.get("status")
            if st == "duplicate" and sj.get("duplicate_of"):
                sj = job_status(str(sj["duplicate_of"]))
                st = sj.get("status")
            if not _terminal(st):
                continue
            try:
                git("add", "-A", "--", rel(archive_job(p, h["slug"], job, sj)))
            except Exception as e:
                print(f"warning: archive of {h['slug']} submit failed: {e}", file=sys.stderr)
            r = sj.get("result") or {}
            name = h.get("promotion_attempt")
            if st == "done" and r.get("passed"):
                s, ci = r.get("speedup"), r.get("speedup_ci")
                rj = rel(os.path.join(hdir(p, "done", d), f"{h['slug']}.result.json"))
                save_best(p, {"problem": p, "job": job, "check_job": h.get("job"), "mode": "submit", "speedup": s, "ci": ci,
                              "score_us": r.get("score_us"), "commit": h.get("promotion_commit"), "submission_blob": _master_blob(p),
                              "slug": h["slug"], "result": rj, "updated_at": ts()})
                you = (okc_json("leaderboard", p) or {}).get("you")
                fm_set(path, promotion="done", board=json.dumps(you) if you else None)
                res_line = f"Promotion passed: {s:.4f}x [{ci[0]:.4f},{ci[1]:.4f}], board: {json.dumps(you)}"
                done.append({"problem": p, "slug": h["slug"], "job": job, "promotion": "done", "speedup": s, "you": you})
            else:
                if h.get("promotion_commit"):
                    git("revert", "--no-edit", h["promotion_commit"], check=False)
                fm_set(path, promotion="unmerged", promotion_reason=f"scored run {st}: {str(r.get('error') or '')[:80]}")
                res_line = f"Promotion run {job} {st}; promotion commit reverted."
                done.append({"problem": p, "slug": h["slug"], "job": job, "promotion": "unmerged"})
            if name and os.path.exists(os.path.join(REPO, p, "attempts", name)):
                ap = os.path.join(REPO, p, "attempts", name)
                txt = re.sub(r"## Result \(after the run\)\n.*\Z", "## Result (after the run)\n" + res_line + "\n", open(ap).read(), flags=re.S)
                open(ap, "w").write(txt)
                git("add", "--", f"{p}/attempts/{name}")
            git("add", "-A", "--", rel(hdir(p)))
            git("commit", "-q", "-m", f"churn({p}): promotion of {h['slug']} → {read_hyp(path).get('promotion')}", check=False)
    out({"ok": True, "promotions": done}, 0 if done else 1)


def _eligible_queue(p):
    items = []
    for f in list_md(hdir(p, "queue")):
        h = read_hyp(os.path.join(hdir(p, "queue"), f))
        if h.get("status") != "queued":
            continue
        if h.get("depends_on"):
            dp, where = find_hyp(p, str(h["depends_on"]))
            if not dp:
                continue
            parent = read_hyp(dp)
            if str(h.get("kind") or "") == "rewrite" and str(parent.get("kind") or "") == "rewrite":
                # a rewrite's next stage builds on the previous stage's code whatever its score:
                # a slow or failing first cut is the expected start of a rewrite, not a dead end
                if not parent.get("impl_commit") or where == "done/rejected" or not where.startswith("done/"):
                    continue
            elif str(h.get("kind") or "").startswith("ab-"):
                # an A/B companion only needs its parent's code to exist; a regressed parent is
                # exactly what it is there to explain
                if not parent.get("impl_commit") or where in ("done/failed", "done/rejected"):
                    continue
            else:
                if not where.startswith("done/") or where in ("done/regression", "done/failed", "done/rejected"):
                    continue
        items.append(h)
    # rewrites are the bigger bets: they go ahead of primaries at the same priority or below
    items.sort(key=lambda h: (0 if h.get("kind") == "rewrite" else 1, h.get("priority") or 3, h.get("created") or ""))
    return items


def checks_today_by_kind(p, t=None):
    """Checks dispatched this UTC day, as (rewrites, everything else), from running and done files."""
    day = (t or now()).strftime("%Y-%m-%d")
    r = o = 0
    dirs = [hdir(p, "running")] + [os.path.join(hdir(p, "done"), d) for d in os.listdir(hdir(p, "done"))
                                   if os.path.isdir(os.path.join(hdir(p, "done"), d))]
    for d in dirs:
        for f in list_md(d):
            h = read_hyp(os.path.join(d, f))
            if h.get("mode") != "check" or str(h.get("dispatched_at") or "")[:10] != day:
                continue
            if h.get("kind") == "rewrite":
                r += 1
            else:
                o += 1
    return r, o


def budget_for(p, q, t=None):
    lim = q.get(p) or {}
    Lc = ((lim.get("checks") or {}).get("left")) or 0
    Ls = ((lim.get("scored") or {}).get("left")) or 0
    H = hours_to_reset(t)
    running = running_hyps(p)
    inflight = [h for h in running if h.get("stage") == "dispatched" and h.get("job") and not h.get("stale")]
    Fc = sum(1 for h in inflight if h.get("mode") == "check")
    F = len(inflight) + sum(1 for h in running if h.get("stage") in ("claimed", "implemented"))
    gains = [f for f in list_md(hdir(p, "done", "gain")) if read_hyp(os.path.join(hdir(p, "done", "gain"), f)).get("promotion") in (None, "pending", "merged")]
    G = len(gains)
    Ps = 0 if H <= 2 else max(G, 1 if Fc > 0 else 0)
    L = Lc + max(0, Ls - Ps)
    Q = _eligible_queue(p)
    need = L
    pace = need if H <= 2 else math.ceil(need / max(H - 1, 0.5))
    N = max(0, min(L, len(Q), PER_TICK_CAP, MAX_IN_FLIGHT - F, pace))
    # the 40% rule: REWRITE_SHARE of the day's checks goes to rewrites, the rest to everything else.
    # Each side's unused share is released to the other when it has nothing eligible, and in the
    # last 2 hours, so the split never leaves a check unspent.
    per_day = ((lim.get("checks") or {}).get("per_day")) or 0
    r_used, o_used = checks_today_by_kind(p, t)
    r_share = math.ceil(REWRITE_SHARE * per_day)
    r_left, o_left = max(0, r_share - r_used), max(0, per_day - r_share - o_used)
    has_r = any(h.get("kind") == "rewrite" for h in Q)
    has_o = any(h.get("kind") != "rewrite" for h in Q)
    if H <= 2 or not has_o:
        r_left = Lc
    if H <= 2 or not has_r:
        o_left = Lc
    picks, lc = [], Lc
    for h in Q:
        if len(picks) >= N:
            break
        rw = h.get("kind") == "rewrite"
        if lc > 0 and (r_left if rw else o_left) > 0:
            picks.append((h, "check"))
            lc -= 1
            if rw:
                r_left -= 1
            else:
                o_left -= 1
        elif h.get("risk") != "high" and not rw:
            picks.append((h, "submit"))
        # a rewrite or high-risk pick with no check in its share waits for one
    ticks_left = max(1, int((H - 0.25) * 3600 / 900))
    projected = min(need, len(Q), pace * ticks_left)
    shortfall = max(0, need - projected)
    if need > 0:
        wake = (H - 1) * 3600 / max(need, 1)
    else:
        wake = 3600
    if F and not picks:
        wake = min(wake, 1800)
    if H < 2:
        wake = min(wake, 600)
    if H > 23.9:
        wake = 300
    wake = int(max(300, min(3600, wake)))
    reason = None
    if N < min(L, pace) and len(Q) <= N:
        reason = f"queue has {len(Q)} eligible hypotheses; {max(0, need - len(Q))} more could be absorbed today (run /hypothesize {p})"
    elif MAX_IN_FLIGHT - F <= 0:
        reason = f"{F} in flight (cap {MAX_IN_FLIGHT})"
    return {"problem": p, "checks_left": Lc, "scored_left": Ls, "hours_to_reset": round(H, 2), "in_flight": F, "in_flight_checks": Fc,
            "gains_pending": G, "promotion_hold": Ps, "spendable": L, "queue": len(Q), "need": need, "pace": pace,
            "N": len(picks), "picks": [{"slug": h["slug"], "mode": m, "priority": h.get("priority"), "review": h.get("review"), "kind": h.get("kind"), "controls_for": h.get("controls_for")} for h, m in picks],
            "shortfall": shortfall, "reason": reason, "wake_s": wake}


def cmd_budget(a):
    t = dt.datetime.fromisoformat(a.now).replace(tzinfo=dt.timezone.utc) if a.now else None
    if a.quotas:
        q = json.load(open(a.quotas))
    else:
        q, _ = quotas()
    plans = [budget_for(p, q, t) for p in ([problem_of(a.problem)] if a.problem else PROBLEMS)]
    total = sum(x["N"] for x in plans)
    if total > GLOBAL_TICK_CAP:
        # trim round-robin from the largest
        while sum(x["N"] for x in plans) > GLOBAL_TICK_CAP:
            big = max(plans, key=lambda x: x["N"])
            big["picks"].pop()
            big["N"] -= 1
    wake = min(x["wake_s"] for x in plans) if plans else 3600
    out({"ok": True, "plans": plans, "wake_s": wake, "dispatch_total": sum(x["N"] for x in plans)})


def log_decision(p, slug, text):
    """Append one line to hypotheses/DECISIONS.md: every hold, deferral and judgement call, with why."""
    path = os.path.join(REPO, "hypotheses", "DECISIONS.md")
    if not os.path.exists(path):
        open(path, "w").write("# Decisions\n\nAppend-only log of churnsubs judgement calls (holds, deferrals, merges by hand,\n"
                              "overrides of a reviewer) with the reason, newest last. Written by `churn.py note` and `hold`.\n\n")
    with open(path, "a") as f:
        f.write(f"- {ts()} `{p}` **{slug}**: {' '.join(str(text).split())}\n")
    git("add", "--", rel(path))


def cmd_archive_file(a):
    """Copy a file with reproducibility value (an implementer brief, a mint prompt) into the archive."""
    p = problem_of(a.problem)
    adir = hdir(p, "archive", a.slug)
    os.makedirs(adir, exist_ok=True)
    dst = os.path.join(adir, a.name or f"{ts()}.{os.path.basename(a.file)}")
    shutil.copyfile(a.file, dst)
    git("add", "--", rel(dst))
    git("commit", "-q", "-m", f"churn({p}): archive {os.path.basename(dst)} for {a.slug}", check=False)
    out({"ok": True, "path": rel(dst)})


def cmd_note(a):
    p = problem_of(a.problem)
    log_decision(p, a.slug, a.text)
    git("commit", "-q", "-m", f"churn({p}): note {a.slug}", check=False)
    out({"ok": True})


def cmd_hold(a):
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path or where != "queue":
        refuse(f"{a.slug} is not in {p}/hypotheses/queue")
    fm_set(path, status="hold" if not a.unhold else "queued", hold_reason=None if a.unhold else (a.reason or "held"))
    log_decision(p, a.slug, ("unhold: " if a.unhold else "hold: ") + (a.reason or ""))
    git("add", "--", rel(path))
    git("commit", "-q", "-m", f"churn({p}): {'unhold' if a.unhold else 'hold'} {a.slug}" + (f": {a.reason}" if a.reason and not a.unhold else ""), check=False)
    out({"ok": True, "slug": a.slug, "status": "queued" if a.unhold else "hold"})


def cmd_claim(a):
    p = problem_of(a.problem)
    if a.slug:
        path, where = find_hyp(p, a.slug)
        if not path or where != "queue":
            refuse(f"{a.slug} is not queued in {p}")
        h = read_hyp(path)
        if h.get("status") != "queued":
            refuse(f"{a.slug} status is {h.get('status')}")
    else:
        Q = _eligible_queue(p)
        if not Q:
            out({"ok": False, "reason": "no eligible hypothesis"}, 1)
        h = Q[0]
    slug = h["slug"]
    errs, _ = lint_hyp(h["_path"], p)
    if errs:
        refuse(f"{slug} fails lint-hyp: {errs}")
    if h.get("review") != "proceed" and not a.skip_review:
        refuse(f"{slug} has not passed the adversarial review (review={h.get('review')}); run the review and "
               f"`churn.py review-record` first, or pass --skip-review deliberately", review=h.get("review"))
    base_ref = "master"
    if h.get("depends_on"):
        dp, _ = find_hyp(p, str(h["depends_on"]))
        parent = read_hyp(dp) if dp else {}
        # A parent already on master builds on master, so later landings are not lost.
        if parent.get("promotion") not in ("merged", "merged-unsubmitted", "submitted", "done", "onboard"):
            base_ref = parent.get("impl_commit") or f"hyp/{p}/{h['depends_on']}"
    elif h.get("base") and h["base"] != "master":
        base_ref = str(h["base"])
    base_sha = git("rev-parse", base_ref).strip()
    base_blob = git("rev-parse", f"{base_sha}:{p}/submission.py").strip()
    br = f"hyp/{p}/{slug}"
    wt = os.path.join(WORKTREES, p, slug)
    new_md = os.path.join(hdir(p, "running"), os.path.basename(h["_path"]))
    git("mv", rel(h["_path"]), rel(new_md))
    fm_set(new_md, status="running", stage="claimed", claimed_at=ts(), base_sha=base_sha, base_blob=base_blob, branch=br,
           worktree=rel(wt), mode=a.mode or None)
    git("add", "--", rel(hdir(p, "running")), rel(hdir(p, "queue")))
    git("commit", "-q", "-m", f"churn({p}): claim {slug} (base {base_sha[:10]})")
    if br in git("branch", "--list", br, "--format=%(refname:short)").split():
        git("worktree", "add", wt, br)
    else:
        git("worktree", "add", "-b", br, wt, base_sha)
    hh = read_hyp(new_md)
    out({"ok": True, "problem": p, "slug": slug, "md": rel(new_md), "worktree": wt, "branch": br, "base_sha": base_sha,
         "submission": os.path.join(wt, p, "submission.py"), "spec": {k: hh.get(k) for k in ("area", "change", "expected", "risk", "depends_on", "priority")},
         "findings": rel(hdir(p, "FINDINGS.md")), "best": load_best(p), "landed": landed(p),
         "review": hh.get("review"), "review_companions": hh.get("review_companions")})


def landed(p):
    """Hypotheses whose code is in master's submission.py lineage, oldest first, with their patches."""
    res = []
    for o in OUTCOMES:
        for f in list_md(hdir(p, "done", o)):
            h = read_hyp(os.path.join(hdir(p, "done", o), f))
            pc = h.get("promotion_commit")
            if not pc:
                continue
            if subprocess.run(["git", "merge-base", "--is-ancestor", str(pc), "HEAD"], cwd=REPO).returncode == 0:
                res.append({"slug": h.get("slug"), "area": h.get("area"), "outcome": o, "speedup": h.get("speedup"),
                            "promotion_commit": pc, "patch": rel(os.path.join(hdir(p, "done", o), f"{h['slug']}.patch")),
                            "md": rel(h["_path"]), "change": str(h.get("change") or "")[:300]})
    res.sort(key=lambda x: git("rev-list", "--count", x["promotion_commit"]).strip())
    return res


def cmd_landed(a):
    out({"ok": True, "problem": problem_of(a.problem), "landed": landed(problem_of(a.problem))})


def cmd_review_record(a):
    """Record an adversarial review verdict on a queued hypothesis; queue its A/B companions."""
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path or where not in ("queue", "running"):
        refuse(f"{a.slug} is not queued or running in {p}")
    h = read_hyp(path)
    v = json.load(open(a.json_file)) if a.json_file != "-" else json.load(sys.stdin)
    adir = hdir(p, "archive", a.slug)
    os.makedirs(adir, exist_ok=True)
    stamp = ts()
    json.dump(v, open(os.path.join(adir, f"review.{stamp}.json"), "w"), indent=1)
    if getattr(a, "prompt", None) and os.path.exists(a.prompt):
        shutil.copyfile(a.prompt, os.path.join(adir, f"review-prompt.{stamp}.md"))
    git("add", "--", rel(adir))
    verdict = v.get("verdict")
    if verdict not in ("proceed", "hold", "proceed-with-companions"):
        refuse("verdict must be proceed | hold | proceed-with-companions")
    created = []
    hyp_py = os.path.join(HERE, "..", "..", "hypothesize", "bin", "hyp.py")
    for c in v.get("companions") or []:
        kind = c.get("kind") or "ab-revert"
        if kind not in ("ab-revert", "ab-tune", "control"):
            refuse(f"companion kind {kind} must be ab-revert | ab-tune | control")
        cmd = [sys.executable, "-I", hyp_py, "new", "--problem", p, "--slug", c["slug"], "--area", c.get("area") or h.get("area") or "?",
               "--change", c["change"], "--risk", c.get("risk") or h.get("risk") or "medium",
               "--priority", str(c.get("priority") or h.get("priority") or 3), "--kind", kind, "--controls-for", a.slug,
               "--depends-on", c.get("depends_on") or a.slug, "--notes", c.get("reason") or "A/B companion from the adversarial review."]
        if c.get("expected"):
            cmd += ["--expected", json.dumps(c["expected"])]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        try:
            j = json.loads(r.stdout or "{}")
        except json.JSONDecodeError:
            j = {"ok": False, "raw": (r.stdout + r.stderr)[-400:]}
        if not j.get("ok"):
            refuse(f"companion {c.get('slug')} could not be written: {j}")
        created.append(j["slug"])
    lines = [f"Reviewed {ts()} by an adversarial subagent (model {v.get('model') or 'opus'}).", "",
             f"**Verdict: {verdict}.** {v.get('summary') or ''}", "",
             "### Justification", v.get("justification") or "(none given)", "",
             "### Confounders on the current kernel"] + ([f"- {x}" for x in (v.get("confounders") or [])] or ["- none identified"]) + ["",
             "### Interference and A/B needs", v.get("interference") or "(none)", ""]
    if created:
        lines += ["### Companions queued"] + [f"- `{c}`: {next((x.get('reason') for x in v.get('companions') or [] if x.get('slug') == c or c.startswith(x.get('slug', '') + '-')), '')}" for c in created] + [""]
    if v.get("open_questions"):
        lines += ["### Open questions"] + [f"- {x}" for x in v["open_questions"]] + [""]
    set_section(path, "Adversarial review", "\n".join(lines))
    upd = {"review": "hold" if verdict == "hold" else "proceed", "review_at": ts(), "review_companions": created or None}
    if verdict == "hold":
        upd.update(status="hold", hold_reason=f"adversarial review: {(v.get('summary') or '')[:160]}")
    fm_set(path, **upd)
    git("add", "--", rel(path), *[f"{p}/hypotheses/queue" for _ in [0]])
    git("commit", "-q", "-m", f"churn({p}): adversarial review of {a.slug} → {verdict}" + (f" (+{len(created)} companions)" if created else ""), check=False)
    out({"ok": True, "slug": a.slug, "verdict": verdict, "status": "hold" if verdict == "hold" else h.get("status"), "companions": created})


def cmd_commit_impl(a):
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path or where != "running":
        refuse(f"{a.slug} is not running in {p}")
    h = read_hyp(path)
    wt = os.path.join(WORKTREES, p, a.slug)
    sub = os.path.join(wt, p, "submission.py")
    if not os.path.exists(sub):
        refuse(f"{sub} missing")
    body = h["_body"]
    rc = get_section(body, "Rules check")
    impl = get_section(body, "Implementation notes")
    if len(impl) < 40:
        refuse("## Implementation notes is empty in the hypothesis file; write what you changed first")
    if rc.count("[x]") < 7 or "[ ]" in rc:
        refuse("## Rules check must have every box ticked ([x]) in the hypothesis file before a run")
    lint = subprocess.run([sys.executable, "-I", LINT, sub, "--problem", p, "--json"], capture_output=True, text=True)
    lj = json.loads(lint.stdout or "{}")
    if lint.returncode != 0:
        _reject_local(p, h, "lint:" + ",".join(sorted({f["rule"] for f in lj.get("findings", []) if f["level"] == "error"})), lj)
    digest = sha256(sub)
    if git("status", "--porcelain", "--", f"{p}/submission.py", cwd=wt).strip():
        git("add", "--", f"{p}/submission.py", cwd=wt)
        git("commit", "-q", "-m", f"impl: {a.slug}", cwd=wt)
    impl_commit = git("rev-parse", "HEAD", cwd=wt).strip()
    blob = git("rev-parse", f"HEAD:{p}/submission.py", cwd=wt).strip()
    if blob == h.get("base_blob"):
        _reject_local(p, h, "no-change: submission.py is identical to its base", lj)
    # local duplicate: identical to a known submission, or to SGLang's baseline.py once prose is
    # stripped (the judge only dedups exact bytes)
    ndigest = norm_sha256(sub)
    for x in all_hyps(p):
        if x.get("slug") != a.slug and x.get("job") and (x.get("submission_sha256") == digest or (ndigest and x.get("norm_sha256") == ndigest)):
            _reject_local(p, h, f"duplicate-local:{x['slug']}", lj, copy_from=x)
    bl = os.path.join(REPO, p, "baseline.py")
    if ndigest and os.path.exists(bl) and norm_sha256(bl) == ndigest and str(h.get("kind") or "primary") != "control":
        base_jobs = [j for j, v in local_jobs().items() if (v.get("file") or "").endswith(f"/{p}/baseline.py")]
        _reject_local(p, h, f"identical-to-baseline (prose aside); baseline jobs: {','.join(base_jobs) or 'none'}", lj)
    fm_set(path, stage="implemented", implemented_at=ts(), impl_commit=impl_commit, submission_sha256=digest, norm_sha256=ndigest, submission_blob=blob,
           lint_warnings=len([f for f in lj.get("findings", []) if f["level"] == "warn"]))
    git("add", "--", rel(path))
    git("commit", "-q", "-m", f"churn({p}): {a.slug} implemented ({impl_commit[:10]})")
    out({"ok": True, "slug": a.slug, "impl_commit": impl_commit, "sha256": digest, "lint": lj})


def _reject_local(p, h, detail, lint_json, copy_from=None):
    slug = h["slug"]
    ddir = hdir(p, "done", "rejected")
    new_md = os.path.join(ddir, os.path.basename(h["_path"]))
    git("mv", rel(h["_path"]), rel(new_md))
    wt = os.path.join(WORKTREES, p, slug)
    patch = git("diff", h.get("base_sha") or "HEAD", "--", f"{p}/submission.py", cwd=wt, check=False) if os.path.isdir(wt) else ""
    open(os.path.join(ddir, f"{slug}.patch"), "w").write(patch)
    res = {"schema": 1, "problem": p, "slug": slug, "job": None, "mode": None, "status": "rejected-local", "passed": None,
           "lint": lint_json, "detail": detail}
    if copy_from:
        src = os.path.join(hdir(p, "done", copy_from.get("outcome") or "neutral"), f"{copy_from['slug']}.result.json")
        if os.path.exists(src):
            res["duplicate_result"] = rel(src)
    json.dump(res, open(os.path.join(ddir, f"{slug}.result.json"), "w"), indent=1)
    set_section(new_md, "Result", f"- **outcome: rejected** ({detail}); no run was spent.\n- lint: {json.dumps(lint_json.get('findings', []))[:1200]}")
    set_section(new_md, "Follow-ups", "- fix the flagged construct and mint a new hypothesis; this file is kept as the record.")
    fm_set(new_md, stage="harvested", status="done", outcome="rejected", verdict_detail=detail, harvested_at=ts())
    git("add", "-A", "--", rel(ddir), rel(hdir(p, "running")))
    git("commit", "-q", "-m", f"churn({p}): {slug} → rejected ({detail[:60]})")
    if os.path.isdir(wt):
        git("worktree", "remove", "--force", wt, check=False)
    out({"ok": False, "slug": slug, "outcome": "rejected", "detail": detail, "lint": lint_json}, 1)


def cmd_prepare_run(a):
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path or where != "running":
        refuse(f"{a.slug} is not running in {p}")
    h = read_hyp(path)
    if h.get("stage") != "implemented":
        refuse(f"{a.slug} stage is {h.get('stage')}, expected implemented")
    if h.get("job"):
        refuse(f"{a.slug} already has job {h['job']}")
    mode = a.mode or h.get("mode")
    if not mode:
        q, _ = quotas()
        lim = q.get(p) or {}
        mode = "check" if ((lim.get("checks") or {}).get("left") or 0) > 0 else "submit"
        if mode == "submit" and h.get("risk") == "high":
            refuse("no checks left and the hypothesis is high risk; not spending a scored run on it")
    wt = os.path.join(WORKTREES, p, a.slug)
    # drop any stale placeholder attempt file for this slug on the branch
    ad = os.path.join(wt, p, "attempts")
    for f in os.listdir(ad) if os.path.isdir(ad) else []:
        if f.endswith(f"-{a.slug}.md") and "`<job id>`" in open(os.path.join(ad, f)).read():
            git("rm", "-q", "--", f"{p}/attempts/{f}", cwd=wt, check=False)
            if os.path.exists(os.path.join(ad, f)):
                os.remove(os.path.join(ad, f))
    body = h["_body"]
    name = write_attempt(wt, p, a.slug, mode, h, get_section(body, "Implementation notes"), get_section(body, "Rules check"),
                         str(h.get("risk") or "") + "\n" + get_section(body, "Risk"), str(h.get("change") or ""))
    git("add", "--", f"{p}/attempts/{name}", cwd=wt)
    git("commit", "-q", "-m", f"attempt: {a.slug} ({mode})", cwd=wt)
    fm_set(path, attempt_file=name, mode=mode, prepared_at=ts())
    git("add", "--", rel(path))
    git("commit", "-q", "-m", f"churn({p}): {a.slug} ready to run ({mode})")
    out({"ok": True, "slug": a.slug, "mode": mode, "attempt_file": name,
         "run": f"okc {mode} {p} {os.path.join(wt, p, 'submission.py')} --no-wait --json",
         "then": f"churn.py record {p} {a.slug} '<okc json or job id>'"})


def cmd_record(a):
    p = problem_of(a.problem)
    path, where = find_hyp(p, a.slug)
    if not path or where != "running":
        refuse(f"{a.slug} is not running in {p}")
    h = read_hyp(path)
    job = _job_from(a.job)
    _record(p, h, job)
    out({"ok": True, "slug": a.slug, "job": job})


def cmd_best_init(a):
    """Seed BEST.json per problem from finished runs of master-lineage submission blobs."""
    lj = local_jobs()
    remote = {j.get("job_id"): j for j in jobs_list() if isinstance(j, dict)}
    res = {}
    for p in PROBLEMS:
        cur = load_best(p)
        if cur and not a.rescan:
            res[p] = {"kept": cur.get("speedup"), "job": cur.get("job")}
            continue
        lineage = set(git("log", "--first-parent", "--format=%H", "--", f"{p}/submission.py").split())
        blobs = {c: git("rev-parse", f"{c}:{p}/submission.py").strip() for c in lineage}
        best = None
        for job, v in lj.items():
            f = v.get("file") or ""
            if not f.endswith(f"/{p}/submission.py") or ".worktrees/" in f:
                continue
            rj = remote.get(job) or {}
            if rj.get("status") != "done" or not rj.get("passed"):
                continue
            sj = job_status(job)
            r = sj.get("result") or {}
            if not r.get("passed") or r.get("speedup") is None:
                continue
            # which lineage blob was it? the file content at submit time is unknown; use content_hash
            # match is impossible, so attribute to the newest lineage commit at or before submit time
            at = v.get("at", 0)
            cand = None
            for c in git("log", "--first-parent", "--format=%H %ct", "--", f"{p}/submission.py").splitlines():
                sha, ct = c.split()
                if int(ct) <= at:
                    cand = sha
                    break
            if not cand:
                cand = sorted(lineage)[0] if lineage else None
            if best is None or r["speedup"] > best["speedup"]:
                best = {"problem": p, "job": job, "mode": rj.get("mode"), "speedup": r["speedup"], "ci": r.get("speedup_ci"),
                        "score_us": r.get("score_us"), "commit": cand, "submission_blob": blobs.get(cand), "slug": None,
                        "result": None, "updated_at": ts(), "note": "seeded by best-init from local job records"}
        if best:
            # keep a trimmed result beside it so FINDINGS can show where the time goes
            sj = job_status(best["job"])
            tr = trim_result(best["job"], sj, p, {"slug": None, "mode": best.get("mode"), "submission_blob": best.get("submission_blob")})
            rp = hdir(p, "BEST.result.json")
            json.dump(tr, open(rp, "w"), indent=1)
            best["result"] = rel(rp)
            save_best(p, best)
            res[p] = {"seeded": best["speedup"], "job": best["job"], "commit": (best.get("commit") or "")[:10]}
        else:
            res[p] = {"seeded": None}
    out({"ok": True, "best": res})


def cmd_index(a):
    changed = []
    roll = ["# Hypotheses across problems", "", f"Regenerated {ts()} by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.", ""]
    try:
        q, _ = quotas()
    except Exception:
        q = {}
    roll += ["| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PROBLEMS:
        hs = all_hyps(p)
        best = load_best(p)
        cnt = lambda w: sum(1 for h in hs if h["_where"] == w)
        lim = q.get(p) or {}
        roll.append(f"| {p} | {('%.4fx' % best['speedup']) if best else '—'} | {cnt('queue')} | {cnt('running')} | {cnt('done/gain')} | {cnt('done/neutral')} | {cnt('done/regression')} | {cnt('done/failed')} | {cnt('done/rejected')} | {((lim.get('checks') or {}).get('left', '?'))} | {((lim.get('scored') or {}).get('left', '?'))} |")
        # per-problem INDEX
        lines = [f"# {p} hypotheses", "", f"Regenerated {ts()}. Best: " + (f"{best['speedup']:.4f}x [{best['ci'][0]:.4f},{best['ci'][1]:.4f}] job `{best['job']}` ({best.get('mode')})" if best else "none yet"), "",
                 "| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for h in sorted(hs, key=lambda h: str(h.get("created") or "")):
            ci = h.get("speedup_ci")
            cis = f"[{ci[0]:.3f},{ci[1]:.3f}]" if isinstance(ci, list) and len(ci) == 2 and ci[0] is not None else ""
            hit = ""
            rjp = os.path.join(hdir(p, h["_where"]), f"{h.get('slug')}.result.json") if h["_where"].startswith("done/") else None
            if rjp and os.path.exists(rjp):
                try:
                    hit = (json.load(open(rjp)).get("attribution") or {}).get("expected_hit") or ""
                except Exception:
                    pass
            kind = (h.get("kind") or "") + (f"→{h['controls_for']}" if h.get("controls_for") else "")
            lines.append(f"| {h.get('created')} | {h.get('slug')} | {kind} | {h.get('area')} | {h['_where']} | {h.get('outcome') or h.get('status')} | {('%.4f' % h['speedup']) if isinstance(h.get('speedup'), (int, float)) else ''} | {cis} | {h.get('job') or ''} | {h.get('mode') or ''} | {h.get('promotion') or ''} | {h.get('review') or ''} | {hit} |")
        held = [h for h in hs if h.get("status") == "hold"]
        if held:
            lines += ["", "## Held", ""] + [f"- {h['slug']}: {h.get('hold_reason')}" for h in held]
        nh = [h for h in hs if h.get("needs_human")]
        if nh:
            lines += ["", "## Needs a human", ""] + [f"- {h['slug']} ({h['_where']}, stage {h.get('stage')})" for h in nh]
        _write_if_changed(hdir(p, "INDEX.md"), "\n".join(lines) + "\n", changed)
        _write_if_changed(hdir(p, "FINDINGS.md"), _findings(p, hs, best), changed)
    try:
        b = json.loads(subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "budget"], capture_output=True, text=True, cwd=REPO).stdout or "{}")
        roll += ["", "## Today's budget", "", "| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |", "|---|---|---|---|---|---|---|---|"]
        for x in b.get("plans", []):
            roll.append(f"| {x['problem']} | {x['checks_left']} | {x['scored_left']} | {x['in_flight']} | {x['queue']} | {x['N']} | {x['shortfall']} | {x.get('reason') or ''} |")
    except Exception as e:
        roll.append(f"\n(budget unavailable: {e})")
    os.makedirs(os.path.join(REPO, "hypotheses"), exist_ok=True)
    _write_if_changed(os.path.join(REPO, "hypotheses", "INDEX.md"), "\n".join(roll) + "\n", changed)
    if changed and not a.no_commit:
        git("add", "--", *changed)
        git("commit", "-q", "-m", "churn: regenerate indexes", check=False)
    out({"ok": True, "changed": [rel(c) for c in changed]})


def _write_if_changed(path, text, changed):
    # strip the volatile timestamp when comparing so a no-op regen does not churn git
    def norm(s):
        return re.sub(r"Regenerated \S+", "Regenerated", s)
    old = open(path).read() if os.path.exists(path) else None
    if old is None or norm(old) != norm(text):
        open(path, "w").write(text)
        changed.append(path)


def _findings(p, hs, best):
    L = [f"# {p} findings", "", f"Regenerated {ts()} by `churn.py index`. Read this before writing a hypothesis for {p}.", ""]
    L += ["## Current best", ""]
    if best:
        L.append(f"{best['speedup']:.4f}x [{best['ci'][0]:.4f},{best['ci'][1]:.4f}] from job `{best['job']}` ({best.get('mode')}), master commit `{(best.get('commit') or '')[:10]}`" + (f", hypothesis {best['slug']}" if best.get("slug") else ""))
        rjp = os.path.join(REPO, best["result"]) if best.get("result") else None
        if rjp and os.path.exists(rjp):
            rj = json.load(open(rjp))
            L += ["", "Where the time goes (per scored shape, top kernels):", ""]
            for s in rj.get("shapes", []):
                tk = ", ".join(f"{k[0]} {k[1]}µs {k[2]}%" for k in (s.get("top_kernels") or [])[:3])
                L.append(f"- {s['key']}: yours {s['median_ns']/1000:.1f}µs vs SGLang {s['baseline_ns']/1000:.1f}µs (SoL {s['speed_of_light_ns']/1000:.1f}µs) · {tk}")
    else:
        L.append("none recorded yet (run `churn.py best-init`).")
    by_area = {}
    for h in hs:
        if h["_where"].startswith("done/"):
            by_area.setdefault(str(h.get("area")), []).append(h)
    L += ["", "## By area", ""]
    for area, items in sorted(by_area.items()):
        L.append(f"### {area}")
        for h in sorted(items, key=lambda h: str(h.get("created"))):
            s = h.get("speedup")
            summ = ""
            rjp = os.path.join(hdir(p, h["_where"]), f"{h['slug']}.result.json")
            if os.path.exists(rjp):
                try:
                    summ = (json.load(open(rjp)).get("attribution") or {}).get("summary") or ""
                except Exception:
                    pass
            L.append(f"- **{h['slug']}** → {h.get('outcome')} ({h.get('verdict_detail')})" + (f", {s:.4f}x" if isinstance(s, (int, float)) else "") + (f" — {summ}" if summ else "") + f". {str(h.get('change') or '')[:160]}")
        L.append("")
    fol = []
    for h in hs:
        if h["_where"] in ("done/failed", "done/neutral"):
            fu = get_section(h["_body"], "Follow-ups")
            if fu:
                fol.append(f"- {h['slug']}: " + fu.splitlines()[0].lstrip("- ")[:220])
    L += ["## Open follow-ups", ""] + (fol or ["- none"]) + [""]
    dn = [f"- {h['slug']}: {h.get('verdict_detail')}" for h in hs if h["_where"] == "done/rejected"]
    L += ["## Do not retry (rejected / duplicates)", ""] + (dn or ["- none"]) + [""]
    L += ["## Queue and in flight", ""] + ([f"- {h['_where']}: {h['slug']} [{h.get('area')}] p{h.get('priority')} {h.get('status')}" for h in hs if h["_where"] in ("queue", "running")] or ["- empty"]) + [""]
    return "\n".join(L)


def cmd_classify(a):
    """Offline: classify a saved `okc status` JSON as if it were a hypothesis result."""
    sj = json.load(open(a.status_json))
    h = {"slug": a.slug or "test", "mode": sj.get("mode")}
    res = trim_result(sj.get("id") or "test", sj, a.problem and problem_of(a.problem) or "?", h)
    best = json.load(open(a.best)) if a.best else None
    base = json.load(open(a.base)) if a.base else None
    res["attribution"] = attribute(res, base, h) if res.get("shapes") else None
    oc, det = classify(res, best, base)
    out({"outcome": oc, "detail": det, "speedup": res.get("speedup"), "ci": res.get("speedup_ci"),
         "tests": res["tests"], "attribution_summary": (res.get("attribution") or {}).get("summary"), "shapes": len(res["shapes"]),
         "result": res if a.full else None})


def cmd_tick(a):
    """preflight → intake → recover → harvest → harvest-promotions → promote(plan) → budget → index."""
    me = [sys.executable, "-I", os.path.abspath(__file__)]
    report = {}

    def sub(name, *args, ok_codes=(0, 1)):
        p = subprocess.run(me + [name, *args], capture_output=True, text=True, cwd=REPO)
        try:
            j = json.loads(p.stdout or "{}")
        except json.JSONDecodeError:
            j = {"_raw": p.stdout[-800:]}
        j["_exit"] = p.returncode
        if p.stderr.strip():
            j["_log"] = p.stderr.strip()[-800:]
        report[name] = j
        return j

    pf = sub("preflight")
    if pf.get("_exit") not in (0,):
        out({"ok": False, "report": report}, 2)
    sub("intake")
    rc = sub("recover")
    sub("harvest")
    sub("harvest-promotions")
    pr = sub("promote")
    b = sub("budget")
    sub("index")
    res = {"ok": True, "promote": pr.get("promote", []), "dispatch": [{"problem": x["problem"], "picks": x["picks"]} for x in b.get("plans", []) if x.get("picks")],
           "wake_s": b.get("wake_s", 1800), "needs_human": rc.get("needs_human", []), "report": report}
    try:
        tdir = os.path.join(REPO, "hypotheses", "ticks")
        os.makedirs(tdir, exist_ok=True)
        tp = os.path.join(tdir, f"{ts()}.json")
        json.dump(res, open(tp, "w"), indent=1)
        git("add", "--", rel(tp))
        git("commit", "-q", "-m", "churn: tick log", check=False)
    except Exception as e:
        print(f"warning: tick log failed: {e}", file=sys.stderr)
    out(res, 0)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("preflight").set_defaults(fn=cmd_preflight)
    sp.add_parser("unlock").set_defaults(fn=cmd_unlock)
    sp.add_parser("intake").set_defaults(fn=cmd_intake)
    x = sp.add_parser("lint-hyp"); x.add_argument("file"); x.add_argument("--problem"); x.set_defaults(fn=cmd_lint_hyp)
    sp.add_parser("recover").set_defaults(fn=cmd_recover)
    x = sp.add_parser("harvest"); x.add_argument("--problem"); x.set_defaults(fn=cmd_harvest)
    sp.add_parser("harvest-promotions").set_defaults(fn=cmd_harvest_promotions)
    x = sp.add_parser("promote"); x.add_argument("--problem"); x.set_defaults(fn=cmd_promote)
    x = sp.add_parser("record-submit"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("job"); x.set_defaults(fn=cmd_record_submit)
    x = sp.add_parser("budget"); x.add_argument("--problem"); x.add_argument("--now"); x.add_argument("--quotas"); x.set_defaults(fn=cmd_budget)
    x = sp.add_parser("hold"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("--reason"); x.set_defaults(fn=cmd_hold, unhold=False)
    x = sp.add_parser("unhold"); x.add_argument("problem"); x.add_argument("slug"); x.set_defaults(fn=cmd_hold, unhold=True, reason=None)
    x = sp.add_parser("claim"); x.add_argument("problem"); x.add_argument("slug", nargs="?"); x.add_argument("--mode"); x.add_argument("--skip-review", action="store_true"); x.set_defaults(fn=cmd_claim)
    x = sp.add_parser("landed"); x.add_argument("problem"); x.set_defaults(fn=cmd_landed)
    x = sp.add_parser("review-record"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("json_file"); x.add_argument("--prompt", help="the filled adversary prompt, archived beside the review"); x.set_defaults(fn=cmd_review_record)
    x = sp.add_parser("note"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("text"); x.set_defaults(fn=cmd_note)
    x = sp.add_parser("archive-file"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("file"); x.add_argument("--as", dest="name"); x.set_defaults(fn=cmd_archive_file)
    x = sp.add_parser("commit-impl"); x.add_argument("problem"); x.add_argument("slug"); x.set_defaults(fn=cmd_commit_impl)
    x = sp.add_parser("prepare-run"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("--mode"); x.set_defaults(fn=cmd_prepare_run)
    x = sp.add_parser("record"); x.add_argument("problem"); x.add_argument("slug"); x.add_argument("job"); x.set_defaults(fn=cmd_record)
    x = sp.add_parser("best-init"); x.add_argument("--rescan", action="store_true"); x.set_defaults(fn=cmd_best_init)
    x = sp.add_parser("index"); x.add_argument("--no-commit", action="store_true"); x.set_defaults(fn=cmd_index)
    x = sp.add_parser("classify"); x.add_argument("status_json"); x.add_argument("--problem"); x.add_argument("--best"); x.add_argument("--base"); x.add_argument("--slug"); x.add_argument("--full", action="store_true"); x.set_defaults(fn=cmd_classify)
    sp.add_parser("tick").set_defaults(fn=cmd_tick)
    a = ap.parse_args()
    try:
        a.fn(a)
    except SystemExit:
        raise
    except Exception as e:
        out({"ok": False, "error": f"{type(e).__name__}: {e}"}, 3)


if __name__ == "__main__":
    main()
