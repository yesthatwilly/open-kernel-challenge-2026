#!/usr/bin/env python3
"""hyp.py — write, lint, list and triage hypothesis files for /hypothesize.

  hyp.py new --problem P --slug S --area A --change TEXT --prior "SLUG: what differs"|none
             --target check|diag|h100 --evidence TEXT --gain PCT --p PROB --cost TEXT
             [--decision TEXT] [--instrument TEXT] [--expected JSON] [--risk low|medium|high]
             [--depends-on SLUG] [--priority N] [--notes TEXT] [--user-requested] [--dry-run]
  hyp.py triage [P]          where the headline can still move, per row of the board's own run (cheap)
  hyp.py list [P]            queue / running / done / h100 per problem, one line each
  hyp.py lint FILE           same check churn.py intake applies
  hyp.py context P           what a hypothesis author should read for P (CLAIMS first, campaign, best, value bar)
  hyp.py claims [P] [--tag T] [--match TEXT]   the newest CLAIMS.md line per mechanism (refuted, landed, shelf ...)
  hyp.py h100-result P SLUG --run ID --verdict held|flipped|untested|null --note TEXT [--effect PCT]
             files a node experiment's result: h100/ -> h100/done/, frontmatter, one CLAIMS line ([node-only] or [lead])

The value bar (the user, 2026-10-28: "really really stringent on what it hypothesizes, because i cant
get small ROI for the tokens this skill takes. i need it only producing the most promising leads"):
every hypothesis except the adversary's A/B companions states measured evidence, the headline gain if
it works (--gain, %), the probability it delivers that (--p) and its cost. `new` refuses (exit 3) when
gain < 2.5% for a check (one judge run's margin is ~1.24%), gain < 3% for the lever a diag / h100
question sizes, or gain x p < 1.5%. Only an idea the user described (--user-requested) passes below it.
`new` also refuses (exit 3) a mechanism whose newest CLAIMS.md line is [refuted] or [superseded] (churn.claim_hits:
the claim names the slug, or shares 2+ distinctive words with its slug, area or change identifiers such as s2b / mc2,
or 1 that is also in a slug the claim names) unless --prior says in 40+ characters what differs, and stamps
mint_blob (master's submission.py blob) and best_at_mint (BEST) for churn's freshness gate.

`new` only ever creates one new file and never commits; churnsubs commits it at its next intake.
check and diag go to <P>/hypotheses/queue/; h100 goes to <P>/hypotheses/h100/ with status hold (a node
experiment for the next GPU window; churn never dispatches it). The filename is
<YYYY-MM-DDTHHMMSSZ>-<slug>.md; a colliding slug gets a -2, -3 suffix.
Exit 0 ok, 1 lint failed, 2 usage, 3 below the value bar.
"""
import argparse
import collections
import datetime as dt
import glob
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHURN = os.path.join(HERE, "..", "..", "churnsubs", "bin", "churn.py")
sys.path.insert(0, os.path.dirname(os.path.abspath(CHURN)))
import churn  # noqa: E402  (shares the problem list, aliases, parser and lint)

PROBLEMS, problem_of, REPO = churn.PROBLEMS, churn.problem_of, churn.REPO

VALUE_MIN_GAIN = 2.5   # % headline a check must deliver if it works (one judge run's margin is ~1.24%)
VALUE_MIN_LEVER = 3.0  # % headline of the lever a diag / h100 question sizes or unlocks
VALUE_MIN_EV = 1.5     # % headline, gain x p
FLOOR_MIN_US = 5.0     # practical floor of one call (a launch or two) when the speed of light is below it
COMPANIONS = ("ab-revert", "ab-tune", "control")
LOG_HEAD = 150         # lines of the active campaign's LOG.md a hypothesis author reads (Now block + latest verdicts)
CLAIM_TAGS = ("landed", "lead", "shelf", "unresolved", "node-only", "refuted", "superseded")
CLAIM_RE = churn.CLAIM_RE    # the CLAIMS.md line contract lives in churn.py
CLAIMS_MARK = "<!-- claims below: append-only, newest last -->"
TRANSFER = "per-SM cycles transfer to the judge H200; HBM rates do not"
NUM_RE = re.compile(r"\d(\.\d+)?\s*(us|µs|ms|%|TB/s|GB/s|x\b|×|instr|cycles|regs|KB|MB|B\b)")
SRC_RE = re.compile(r"\b[0-9a-f]{8,16}\b|runs/|research/|FINDINGS|LOG|DIAGNOSIS|LESSONS|profile|anatomy|ncu|nsys|gpunode06|\.md\b")


def yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if re.search(r"[:#\[\]{}\"']|^\s|\s$|^[-?&*!|>%@`]", s) or s == "":
        return json.dumps(s)
    return s


def folded(text):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 96:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return ">\n" + "\n".join("  " + l for l in lines)


def h100_hyps(p, done=False):
    res = []
    for sub in ("h100", "h100/done") if done else ("h100",):
        d = os.path.join(REPO, p, "hypotheses", sub)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith(".md"):
                h = churn.read_hyp(os.path.join(d, f))
                h["_where"] = sub
                res.append(h)
    return res


def scratch_dir():
    """A git-ignored scratch directory in the repo (the node's /tmp is a small, nearly full tmpfs)."""
    d = os.path.join(REPO, "hypotheses", "prompts", "filled")
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, ".gitignore")):
        open(os.path.join(d, ".gitignore"), "w").write("*\n")
    return d


def claims_path(p):
    return os.path.join(REPO, p, "hypotheses", "CLAIMS.md")


def read_claims(p):
    """The newest CLAIMS.md line per mechanism (churn.current_claims: a later line about the same slug, or with the
    same text, supersedes an earlier one)."""
    return churn.current_claims(p)


def matching_claims(p, slug, area, change, tags=("refuted", "superseded"), expected=None):
    """churn.claim_hits over this problem's hypotheses, h100 ones included, on the rows of `expected`: [(claim, why)]."""
    rows = ", ".join(str(e.get("shapes") or "*") for e in expected or [] if isinstance(e, dict)) or None
    return churn.claim_hits(p, slug, area, change, tags, churn.all_hyps(p) + h100_hyps(p, done=True), rows)


def claim_line(tag, mechanism, rows, evidence, board, blob, reopen, source):
    f = lambda v: " ".join(str(v if v is not None else "-").replace("|", "/").split()) or "-"
    line = (f"- [{tag}] {f(mechanism)} | rows: {f(rows)} | evidence: {f(evidence)} | board: {f(board)} | blob: {f(blob)} "
            f"| reopen: {f(reopen)} | {dt.datetime.now(dt.timezone.utc):%Y-%m-%d} {source}")
    assert CLAIM_RE.match(line), line
    return line


CLAIMS_HEADER = """# Claims: {p}

Curated findings for {p}, one line each, written by agents on master in the main checkout and never
regenerated (FINDINGS.md is; this file is not). Readers take the newest line per mechanism: a later line about
the same mechanism supersedes an earlier one. `hyp.py claims {p}` prints that view.

Tags: [landed] on master, owns its rows (lineage) · [lead] moved its rows, >= 2.5% headline-equivalent, not on
master (check it first) · [shelf] moved its rows below the bar (stack only to >= 2.5%) · [unresolved] inside row
noise or a route flipped (re-test only with more runs or pinned routes) · [node-only] H100 only, transfer held /
flipped / untested (a schedule change needs a production-shape judge check) · [refuted] rows flat or lost
(binding Do-not-try) · [superseded] a later landing owns those rows (queued items that depend on it are held).

Line format: `- [tag] <mechanism> | rows: <row keys or 'all'> | evidence: <job ids / commits / H100 run ids, with
source path:line> | board: <board speedup when measured> | blob: <short blob of submission.py, or '-'> |
reopen: <condition, or '-'> | <YYYY-MM-DD> <source: churn|campaign|h100|screen>`

"""


def require_master_main():
    """Claims are written on master in the main checkout: a worktree's copy lands on its branch."""
    gd = subprocess.run(["git", "rev-parse", "--git-dir", "--git-common-dir", "--abbrev-ref", "HEAD"], capture_output=True,
                        text=True, cwd=REPO).stdout.split()
    if len(gd) < 3 or os.path.abspath(os.path.join(REPO, gd[0])) != os.path.abspath(os.path.join(REPO, gd[1])) or gd[2] != "master":
        print(f"claims are written on master in the main checkout, not {REPO} ({' '.join(gd[2:]) or '?'})", file=sys.stderr)
        sys.exit(2)


def append_claim(p, line):
    """Append one line to <p>/hypotheses/CLAIMS.md (created with its header when missing)."""
    require_master_main()
    f = claims_path(p)
    head = getattr(churn, "CLAIMS_HEAD", None)  # churn.py's header when it has one, so both writers agree
    text = open(f).read() if os.path.exists(f) else head.format(p=p) if head else CLAIMS_HEADER.format(p=p) + CLAIMS_MARK + "\n"
    open(f, "w").write(text + ("" if text.endswith("\n") else "\n") + line + "\n")


def active_campaign():
    """campaigns/<dir> of the row of campaigns/INDEX.md whose status says active, or None."""
    f = os.path.join(REPO, "campaigns", "INDEX.md")
    for line in open(f) if os.path.exists(f) else []:
        cells = [c.strip() for c in line.split("|")]
        m = re.match(r"\[[^\]]*\]\(([^)/]+)/", cells[1]) if len(cells) > 3 else None
        if m and re.search(r"\bactive\b", cells[2]) and not re.search(r"\b(closed|inactive)\b", cells[2]):
            return f"campaigns/{m.group(1)}"
    return None


def diagnosis_info(p):
    """DIAGNOSIS.md's date and board against today's: binding only while BEST is within margin() of its board."""
    f = os.path.join(REPO, p, "hypotheses", "DIAGNOSIS.md")
    if not os.path.exists(f):
        return None
    fm = churn.parse_fm(open(f).read())[0]
    m = re.search(r"(\d+\.\d+)\s*x", str(fm.get("best") or ""))
    then, now_ = (float(m.group(1)) if m else None), (churn.load_best(p) or {}).get("speedup")
    moved = abs(now_ / then - 1) if then and now_ else None
    hist = moved is None or moved > churn.margin()
    return {"path": f"{p}/hypotheses/DIAGNOSIS.md", "diagnosed_at": str(fm.get("diagnosed_at")), "board_then": then,
            "board_now": round(now_, 4) if now_ else None, "moved_pct": round(moved * 100, 2) if moved is not None else None,
            "historical": hist,
            "note": ("historical: written on another board; its Actions and Do-not-try are context, not binding. Check each "
                     "against CLAIMS.md and triage before minting it" if hist else
                     "current (board within margin): its Actions come first and its Do-not-try binds")}


def unique_slug(p, slug):
    taken = {h.get("slug") for h in churn.all_hyps(p) + h100_hyps(p, done=True)}
    if slug not in taken:
        return slug
    n = 2
    while f"{slug}-{n}" in taken:
        n += 1
    return f"{slug}-{n}"


def check_prior(p, a):
    """`--prior "<slug>: what differs this time"` (or `none`): the closest earlier attempt, so nobody
    re-mints a measured dead end without saying why this one is different. Companions are exempt."""
    if a.kind in COMPANIONS:
        return a.prior
    if not a.prior:
        print("--prior is required: '<closest earlier slug>: what differs this time', or 'none'", file=sys.stderr)
        sys.exit(2)
    if a.prior.strip().lower() == "none":
        return "none"
    slug, _, differs = a.prior.partition(":")
    path, where = churn.find_hyp(p, slug.strip())
    if not path:
        print(f"--prior {slug.strip()} not found in {p}", file=sys.stderr)
        sys.exit(2)
    h = churn.read_hyp(path)
    dead = where in ("done/regression", "done/rejected") or h.get("verdict_detail") == "within-ci"
    if dead and len(differs.strip()) < 40:
        print(f"--prior {slug.strip()} is a measured dead end ({where}, {h.get('verdict_detail')}): say in 40+ "
              "characters what is different this time", file=sys.stderr)
        sys.exit(2)
    return f"{slug.strip()}: {differs.strip()}" if differs.strip() else slug.strip()


def check_value(a):
    """The value bar. Returns the frontmatter fields, or exits 2 (something missing) / 3 (below the bar)."""
    if a.kind in COMPANIONS:
        return {}
    errs = []
    ev_text = (a.evidence or "").strip()
    if len(ev_text) < 80 or not NUM_RE.search(ev_text) or not SRC_RE.search(ev_text):
        errs.append("--evidence: the measured gap this attacks, 80+ chars, with numbers and units and a source "
                    "(job or run id, profile line, research/ file, FINDINGS or LOG entry)")
    if a.gain is None or a.p is None or not 0 < a.p <= 1:
        errs.append("--gain (headline % if it works; diag / h100: the headline % of the lever it sizes) and "
                    "--p (0..1, the chance it delivers that) are required")
    if not (a.cost or "").strip():
        errs.append("--cost: '<tokens>k tokens, <n> judge runs' (h100: '<n> GPU-min')")
    if a.target in ("diag", "h100") and len((a.decision or "").strip()) < 60:
        errs.append("--decision: the pre-registered rule, 60+ chars: which outcome leads to which build, port or close")
    if a.target == "check" and not a.expected:
        errs.append("--expected: the rows it moves, with the arithmetic behind --gain")
    if errs:
        print(json.dumps({"ok": False, "errors": errs}))
        sys.exit(2)
    ev = round(a.gain * a.p, 2)
    need = VALUE_MIN_LEVER if a.target in ("diag", "h100") else VALUE_MIN_GAIN
    below = ([f"gain {a.gain}% < {need}%"] if a.gain < need else []) + \
            ([f"EV {ev}% < {VALUE_MIN_EV}%"] if ev < VALUE_MIN_EV else [])
    if below and not a.user_requested:
        print(json.dumps({"ok": False, "below_bar": below, "ev": ev,
                          "hint": "do not mint it; report it as a rejected lead with its EV"}))
        sys.exit(3)
    return {"gain": a.gain, "p": a.p, "ev": ev, "need": need, "below": "; ".join(below)}


def cmd_new(a):
    p = problem_of(a.problem)
    slug = re.sub(r"[^a-z0-9-]+", "-", a.slug.lower()).strip("-")[:60]
    if len(slug) < 3:
        print("slug too short", file=sys.stderr)
        sys.exit(2)
    asked, slug = slug, unique_slug(p, slug)
    expected = json.loads(a.expected) if a.expected else []
    if a.depends_on:
        dp, where = churn.find_hyp(p, a.depends_on)
        if not dp:
            print(f"depends_on {a.depends_on} not found in {p}", file=sys.stderr)
            sys.exit(2)
    prior = check_prior(p, a)
    hits = [] if a.kind in COMPANIONS else matching_claims(p, asked, a.area, a.change, expected=expected)
    for c, w in hits:
        print(f"CLAIMS.md:{c['line']} ({', '.join(w)}): {c['text']}", file=sys.stderr)
    if hits and len((a.prior or "").partition(":")[2].strip()) < 40:
        print(json.dumps({"ok": False, "claims": [c["text"] for c, _ in hits],
                          "hint": "the newest claim on this mechanism is refuted/superseded: do not mint it, or say in "
                                  "--prior '<slug>: ...' (40+ chars) what differs, e.g. the claim's reopen condition is met"}))
        sys.exit(3)
    val = check_value(a)
    h100 = a.target == "h100" and a.kind not in COMPANIONS
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    blob = subprocess.run(["git", "rev-parse", f"master:{p}/submission.py"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    best = (churn.load_best(p) or {}).get("speedup")
    fm = ["---", f"problem: {p}", f"slug: {slug}", f"created: {created}", "base: master",
          f"mint_blob: {json.dumps(blob) if blob else 'null'}", f"best_at_mint: {round(best, 4) if best else 'null'}",
          f"area: {yaml_scalar(a.area)}", f"change: {folded(a.change)}"]
    if expected:
        fm.append("expected:")
        for e in expected:
            fm.append(f"  - shapes: {yaml_scalar(e.get('shapes', '*'))}")
            fm.append(f"    direction: {yaml_scalar(e.get('direction', 'faster'))}")
            fm.append(f"    why: {yaml_scalar(e.get('why', ''))}")
    else:
        fm.append("expected: []")
    fm += [f"risk: {a.risk}", f"depends_on: {yaml_scalar(a.depends_on)}", f"priority: {a.priority}",
           f"kind: {a.kind}", f"controls_for: {yaml_scalar(a.controls_for)}", f"prior: {yaml_scalar(prior)}"]
    if a.kind == "rewrite":
        fm += [f"rewrite_stage: {a.stage}", f"replaces: {yaml_scalar(a.replaces or a.area)}"]
    if val:
        fm += [f"target: {a.target}", f"value_gain: {val['gain']}", f"value_p: {val['p']}", f"value_ev: {val['ev']}",
               f"value_cost: {yaml_scalar(a.cost.strip())}", f"evidence: {folded(a.evidence)}"]
        if a.decision:
            fm.append(f"decision: {folded(a.decision)}")
        if a.instrument:
            fm.append(f"instrument: {folded(a.instrument)}")
        if a.user_requested:
            fm.append("user_requested: true")
    if h100:
        fm += ["review: required", "status: hold", 'hold_reason: "h100: waits for a GPU window (the user\'s go)"', "---"]
    else:
        fm += ["review: required", "status: queued", "---"]
    body = [f"# {slug}", "", "## Change", "", a.change.strip(), "", "## Expected", ""]
    body += [f"- `{e.get('shapes', '*')}` {e.get('direction', 'faster')}: {e.get('why', '')}" for e in expected] or ["- (none stated)"]
    if val:
        what = {"check": "a candidate to beat the board in one judge check",
                "diag": "a prepare-only judge diagnostic that sizes a lever",
                "h100": "a node experiment for the next GPU window (the user opens windows; churn never runs it)"}[a.target]
        body += ["", "## Value case", "", f"- **Target:** {a.target}: {what}.",
                 f"- **Evidence:** {a.evidence.strip()}",
                 f"- **Gain if it works:** {val['gain']}% headline; **p** {val['p']}; **EV** {val['ev']}% "
                 f"(bar: gain >= {val['need']}%, EV >= {VALUE_MIN_EV}%)"
                 + (f"; **below the bar** ({val['below']}), minted because the user described it" if val["below"] else "") + ".",
                 f"- **Cost:** {a.cost.strip()}."]
        if a.decision:
            body.append(f"- **Decision rule:** {a.decision.strip()}")
        if a.instrument:
            body.append(f"- **Instrument:** {a.instrument.strip()}")
    body += ["", "## Risk", "", (a.notes or f"{a.risk} risk.").strip(), ""]
    if a.kind == "rewrite":
        body += ["## Milestones", "",
                 f"Rewrite stage {a.stage} of `{a.replaces or a.area}`. Judged on the replaced kernels' own time per",
                 "shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:",
                 "1. correct: passes every test (a failure is a fix for the next stage, not a dead end);",
                 "2. at or below the incumbent kernels' time on the target shapes;",
                 "3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).",
                 "The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.", ""]
    qdir = os.path.join(REPO, p, "hypotheses", "h100" if h100 else "queue")
    name = f"{created}-{slug}.md"
    text = "\n".join(fm) + "\n\n" + "\n".join(body)
    tmp = tempfile.mkdtemp(prefix="dryrun-", dir=scratch_dir()) if a.dry_run else None
    path = os.path.join(tmp or qdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    errs, _ = churn.lint_hyp(path, p)
    if errs or a.dry_run:
        shutil.rmtree(tmp, ignore_errors=True) if tmp else os.remove(path)
    if errs:
        print(json.dumps({"ok": False, "errors": errs}))
        sys.exit(1)
    out = {"ok": True, "file": os.path.relpath(os.path.join(qdir, name), REPO), "slug": slug, "problem": p}
    if hits:
        out["claims"] = [c["text"] for c, _ in hits]
    if val:
        out.update(target=a.target, ev=val["ev"])
    if a.dry_run:
        out["dry_run"] = True
    print(json.dumps(out))


def board_record(p):
    """The board run's okc record: the campaign runs archive, the churn archive, else okc itself."""
    best = churn.load_best(p) or {}
    job = best.get("job")
    if not job:
        return best, None
    for pat in (f"campaigns/*/runs/{job}.json.gz", f"{p}/hypotheses/archive/*/{job}.okc.json.gz"):
        for f in glob.glob(os.path.join(REPO, pat)):
            try:
                return best, json.load(gzip.open(f))
            except (OSError, ValueError):
                pass
    r = subprocess.run(["okc", "status", job, "--json"], capture_output=True, text=True)
    try:
        return best, json.loads(r.stdout)
    except ValueError:
        return best, None


def task_rows(p):
    """(load, shape, weight, floor_us) per row of TASK.md's shape table, in benchmark index order."""
    rows = []
    for line in open(os.path.join(REPO, p, "TASK.md")):
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([\d.,]+)\s*µs\s*\|", line)
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3)), float(m.group(4).replace(",", ""))))
    return rows


def cmd_triage(a):
    probs = [problem_of(a.problem)] if a.problem else PROBLEMS
    cut = time.time() - 3 * 86400
    for p in probs:
        best, rec = board_record(p)
        print(f"== {p}: board {best.get('speedup') or 0:.4f} (job {best.get('job')})")
        bench = sorted([x for x in ((rec or {}).get("result") or {}).get("records", []) if x.get("event") == "benchmark"],
                       key=lambda x: x["index"])
        rows = task_rows(p)
        if bench and len(bench) == len(rows):
            # The score is the geometric mean over loads of (SGLang load time / ours), a load's time being
            # sum(weight x row median_ns) (TASK.md). So a row moves the headline through its own load only:
            # headline factor = (T_load / T_load')^(1/L) with T_load' = T_load - w x (t - t').
            t_of = [x["median_ns"] / 1e3 for x in bench]
            loads = collections.OrderedDict()
            for i, (load, _, w, _) in enumerate(rows):
                loads[load] = loads.get(load, 0.0) + w * t_of[i]
            L = len(loads)
            out, total = [], 1.0
            for i, (x, (load, shape, w, _)) in enumerate(zip(bench, rows)):
                t, b, sol = t_of[i], x["baseline_ns"] / 1e3, x["speed_of_light_ns"] / 1e3
                T = loads[load]
                share = w * t / T / L
                floor = max(1.3 * sol, FLOOR_MIN_US)
                at_floor = (T / (T - w * (t - floor))) ** (1 / L) if t > floor else 1.0
                halved = (T / (T - w * t / 2)) ** (1 / L)
                total *= at_floor
                tag = " ".join(k for k in shape.replace(",", "").split() if k in ("sliding", "full")) or shape[:24]
                out.append((at_floor, f"  #{x['index']:<2} {load:10} {tag:9} w{w:<3} share {share * 100:4.1f}%  ours {t:7.1f} us  "
                                      f"sglang {b:7.1f}  x{x['speedup']:.2f}  SoL {sol:6.1f}  "
                                      f"ours/floor {t / floor:4.1f}  headline x{halved:.3f} if halved, x{at_floor:.3f} at "
                                      f"max(1.3 SoL, {FLOOR_MIN_US:.0f} us)"))
            for _, line in sorted(out, reverse=True):
                print(line)
            print(f"  every row at its practical floor: headline x{total:.2f} (share = the row's elasticity of the headline; "
                  f"median_ns, geomean over {L} loads)")
        else:
            print("  (no per-row data: board record not found or rows do not match TASK.md)")
        q = [h for h in churn.all_hyps(p) if h["_where"] in ("queue", "running")]
        def ev(h):
            try:
                return float(h.get("value_ev"))
            except (TypeError, ValueError):
                return None
        valued = [h for h in q if (ev(h) or 0) >= VALUE_MIN_EV]
        recent = collections.Counter()
        for f in glob.glob(os.path.join(REPO, p, "hypotheses", "done", "*", "*.result.json")):
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue
            if (d.get("created_at") or 0) >= cut:
                recent[f.split(os.sep)[-2]] += 1
        print(f"  queue {len(q)} ({len(valued)} clear the value bar, {len(q) - len(valued)} older files without a value case); "
              f"h100 {len(h100_hyps(p))}; churn results, last 3 days: {dict(recent) or 'none'}")


def cmd_list(a):
    probs = [problem_of(a.problem)] if a.problem else PROBLEMS
    for p in probs:
        for h in churn.all_hyps(p) + h100_hyps(p, done=True):
            s = h.get("speedup")
            ev = f" EV {h['value_ev']}%" if h.get("value_ev") is not None else ""
            print(f"{p} {h['_where']:16} {h.get('slug'):40} [{h.get('area')}] p{h.get('priority')} {h.get('outcome') or h.get('h100_verdict') or h.get('status')}"
                  + (f" {s:.4f}x" if isinstance(s, (int, float)) else "") + ev)


def cmd_lint(a):
    errs, h = churn.lint_hyp(a.file)
    print(json.dumps({"ok": not errs, "errors": errs, "slug": h.get("slug"), "problem": h.get("problem")}))
    sys.exit(1 if errs else 0)


def cmd_context(a):
    p = problem_of(a.problem)
    hd = os.path.join(REPO, p, "hypotheses")
    best = churn.load_best(p)
    diag = diagnosis_info(p)
    camp = active_campaign()
    cfiles = [f for f in (f"{camp}/FINDINGS.md", f"{camp}/DISTILL.md", f"{camp}/research/h100-findings.md")
              if camp and os.path.exists(os.path.join(REPO, f))]
    claims = read_claims(p)
    first = [f"{p}/hypotheses/CLAIMS.md"] if os.path.exists(claims_path(p)) else []
    first += [f"{p}/hypotheses/FINDINGS.md", "hypotheses/LESSONS.md"] + cfiles
    first += [diag["path"]] if diag and not diag["historical"] else []
    print(json.dumps({
        "problem": p,
        "read_first": first + [f"{p}/TASK.md", f"{p}/baseline.py", f"{p}/submission.py", "rules/technical-rules.llms.txt"],
        "claims": {"newest_per_mechanism": dict(collections.Counter(c["tag"] for c in claims)),
                   "check": f"hyp.py claims {p} [--tag refuted] [--match '<mechanism words>']",
                   "rule": "never mint a [refuted]/[superseded] mechanism without saying what differs; rows a [landed] claim "
                           "owns are lineage; mint from [lead] and from [shelf] stacks that sum to >= 2.5%"} if claims
                  else "no CLAIMS.md lines yet",
        "campaign": {"dir": camp, "read": cfiles,
                     "log": f"the top {LOG_HEAD} lines of {camp}/LOG.md (the Now block: boards, in-flight work, the "
                            "latest kills and verdicts); grep the rest for this problem's name only when needed",
                     "focus": "DISTILL §3 (kills, traps, reopen rules) and §4.4 (verdicts on queued items); h100-findings: "
                              f"grep {p.replace('muse_', '')}; {TRANSFER}"} if camp else None,
        "diagnosis": diag,
        "triage": f"hyp.py triage {p}",
        "value_bar": {"check_min_gain_pct": VALUE_MIN_GAIN, "lever_min_gain_pct": VALUE_MIN_LEVER, "min_ev_pct": VALUE_MIN_EV,
                      "judge_margin_pct": round(churn.margin() * 100, 2)},
        "diagnosis_due": p in churn.diagnose_due(),
        "best": best,
        "queue": [h.get("slug") for h in churn.all_hyps(p) if h["_where"] in ("queue", "running")],
        "h100": [h.get("slug") for h in h100_hyps(p)],
        "done": [(h.get("slug"), h.get("outcome")) for h in churn.all_hyps(p) if h["_where"].startswith("done/")],
        "areas_hint": churn.AREAS_HINT,
        "submission_at_master": subprocess.run(["git", "rev-parse", "--short", f"master:{p}/submission.py"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
    }, indent=1))


def cmd_claims(a):
    """The newest claim per mechanism, grouped by tag; --match keeps those whose mechanism shares a word with TEXT."""
    for p in [problem_of(a.problem)] if a.problem else PROBLEMS:
        mw = {w for w in churn.claim_words(a.match or "") - churn.CLAIM_STOP if len(w) >= 3}
        cs = [c for c in read_claims(p) if (not a.tag or c["tag"] in a.tag) and (not a.match or mw & churn.claim_words(c["mechanism"]))]
        print(f"== {p}: {len(cs)} claim(s){' matching ' + repr(a.match) if a.match else ''} ({os.path.relpath(claims_path(p), REPO)})")
        for c in sorted(cs, key=lambda c: (CLAIM_TAGS.index(c["tag"]), c["line"])):
            print(f"  :{c['line']:<4} {c['text']}")


def cmd_h100_result(a):
    """File a node experiment's result: frontmatter, a `## H100 result` section, h100/ -> h100/done/, one CLAIMS line."""
    p = problem_of(a.problem)
    d = os.path.join(REPO, p, "hypotheses", "h100")
    m = [f for f in sorted(os.listdir(d)) if f.endswith(f"-{a.slug}.md")] if os.path.isdir(d) else []
    if not m:
        print(f"no {p}/hypotheses/h100/*-{a.slug}.md (done ones are in h100/done/)", file=sys.stderr)
        sys.exit(2)
    if len(a.note.strip()) < 20:
        print("--note: what was measured, 20+ chars (numbers, node vs judge)", file=sys.stderr)
        sys.exit(2)
    src, dst = os.path.join(d, m[-1]), os.path.join(d, "done", m[-1])
    h = churn.read_hyp(src)
    tag = "lead" if a.verdict == "held" and (a.effect or 0) >= VALUE_MIN_GAIN else "node-only"
    rows = ", ".join(str(e.get("shapes")) for e in h.get("expected") or [] if isinstance(e, dict)) or "all"
    reopen = "-" if tag == "lead" else {
        "flipped": f"only with a production-shape judge check of a changed schedule ({TRANSFER})",
        "held": f"stack it to >= {VALUE_MIN_GAIN}% headline, or re-measure on the judge ({TRANSFER})"}.get(
        a.verdict, f"a production-shape judge check before any build relies on it ({TRANSFER})")
    sec = [f"- **Run:** {a.run}", f"- **Transfer verdict:** {a.verdict}" + (f"; effect {a.effect}% headline" if a.effect is not None else ""),
           f"- **Note:** {a.note.strip()}", f"- **Caveat:** {TRANSFER}.", f"- **Claim:** [{tag}]"]
    mk = lambda n: claim_line(tag, a.mechanism or f"{h.get('area')} ({h.get('slug')})", rows,
                              f"H100 {a.run}, transfer {a.verdict}: {a.note.strip()} ({os.path.relpath(dst, REPO)}:{n})",
                              f"{a.board:.4f}" if a.board else "-", a.blob or "-", reopen, "h100")
    line = mk(1)  # validates the line before anything is written
    if a.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "file": os.path.relpath(dst, REPO), "claim": line}))
        return
    require_master_main()
    churn.fm_set(src, status="done", h100_run=a.run, h100_verdict=a.verdict, h100_effect=a.effect, claim=tag,
                 h100_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"))
    churn.set_section(src, "H100 result", "\n".join(sec))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(src, dst)
    line = mk(open(dst).read().split("\n").index("## H100 result") + 1)
    append_claim(p, line)
    print(json.dumps({"ok": True, "file": os.path.relpath(dst, REPO), "claim": line}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("new")
    x.add_argument("--problem", required=True); x.add_argument("--slug", required=True); x.add_argument("--area", required=True)
    x.add_argument("--change", required=True); x.add_argument("--expected"); x.add_argument("--risk", default="medium", choices=["low", "medium", "high"])
    x.add_argument("--depends-on"); x.add_argument("--priority", type=int, default=3); x.add_argument("--notes")
    x.add_argument("--kind", default="primary", choices=["primary", "rewrite", "ab-revert", "ab-tune", "control"],
                   help="primary = an idea of its own; rewrite = replace a whole subsystem over staged runs; "
                        "ab-revert/ab-tune/control = an A/B companion queued by the adversarial review")
    x.add_argument("--stage", type=int, default=1, help="rewrite only: stage number (1 = first cut)")
    x.add_argument("--replaces", help="rewrite only: the kernels/subsystem being replaced, e.g. 'Marlin w13+w2 at bs<=8'")
    x.add_argument("--controls-for", help="slug of the hypothesis this companion is an A/B control for")
    x.add_argument("--prior", help="'<closest earlier slug>: what differs this time', or 'none' (required except for companions)")
    x.add_argument("--target", default="check", choices=["check", "diag", "h100"],
                   help="check = should beat the board in one judge check; diag = a prepare-only judge diagnostic "
                        "that sizes a lever; h100 = a node experiment for the next GPU window (written to h100/)")
    x.add_argument("--evidence", help="the measured gap: numbers, units and a source (job / run id, profile, research/, FINDINGS)")
    x.add_argument("--gain", type=float, help="headline %% if it works (diag / h100: the headline %% of the lever it sizes)")
    x.add_argument("--p", type=float, help="probability (0..1) that it delivers --gain on the judge")
    x.add_argument("--cost", help="'<tokens>k tokens, <n> judge runs' (h100: '<n> GPU-min')")
    x.add_argument("--decision", help="diag / h100: the pre-registered rule, which outcome leads to which build or close")
    x.add_argument("--instrument", help="h100: how it is measured (nodejudge arm, microbenchmark, ncu / nsys)")
    x.add_argument("--user-requested", action="store_true", help="the user described this idea: it may pass below the bar")
    x.add_argument("--dry-run", action="store_true", help="validate and lint without writing the file")
    x.set_defaults(fn=cmd_new)
    x = sp.add_parser("triage"); x.add_argument("problem", nargs="?"); x.set_defaults(fn=cmd_triage)
    x = sp.add_parser("list"); x.add_argument("problem", nargs="?"); x.set_defaults(fn=cmd_list)
    x = sp.add_parser("lint"); x.add_argument("file"); x.set_defaults(fn=cmd_lint)
    x = sp.add_parser("context"); x.add_argument("problem"); x.set_defaults(fn=cmd_context)
    x = sp.add_parser("claims"); x.add_argument("problem", nargs="?")
    x.add_argument("--tag", action="append", choices=CLAIM_TAGS, help="keep only this tag (repeatable)")
    x.add_argument("--match", help="keep claims whose mechanism shares a word (3+ chars) with TEXT")
    x.set_defaults(fn=cmd_claims)
    x = sp.add_parser("h100-result"); x.add_argument("problem"); x.add_argument("slug")
    x.add_argument("--run", required=True, help="hopper-probe run id or queue job")
    x.add_argument("--verdict", required=True, choices=["held", "flipped", "untested", "null"],
                   help="judge transfer: held, flipped, untested (node only), null (no effect on the node)")
    x.add_argument("--note", required=True, help="what was measured: node numbers and, when checked, the judge's")
    x.add_argument("--effect", type=float, help="headline %% it is worth on the judge (held >= 2.5 files a [lead])")
    x.add_argument("--mechanism", help="claim mechanism (default: '<area> (<slug>)')")
    x.add_argument("--board", type=float, help="the judge board speedup when a judge check measured it")
    x.add_argument("--blob", help="short blob of the submission.py bytes it was measured on")
    x.add_argument("--dry-run", action="store_true", help="print the claim line; change nothing")
    x.set_defaults(fn=cmd_h100_result)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
