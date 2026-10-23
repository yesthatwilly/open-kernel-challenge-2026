#!/usr/bin/env python3
"""hyp.py — write, lint and list hypothesis files for /hypothesize.

  hyp.py new --problem P --slug S --area A --change TEXT [--expected JSON] [--risk low|medium|high]
             [--depends-on SLUG] [--priority N] [--notes TEXT]
  hyp.py list [P]            queue / running / done per problem, one line each
  hyp.py lint FILE           same check churn.py intake applies
  hyp.py context P           what a hypothesis author should read for P (paths + current best)

`new` only ever creates one new file in <P>/hypotheses/queue/ and never commits; churnsubs commits
it at its next intake. The filename is <YYYY-MM-DDTHHMMSSZ>-<slug>.md; a colliding slug gets a -2,
-3 suffix. Exit 0 ok, 1 lint failed, 2 usage.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHURN = os.path.join(HERE, "..", "..", "churnsubs", "bin", "churn.py")
sys.path.insert(0, os.path.dirname(os.path.abspath(CHURN)))
import churn  # noqa: E402  (shares the problem list, aliases, parser and lint)

PROBLEMS, problem_of, REPO = churn.PROBLEMS, churn.problem_of, churn.REPO


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


def unique_slug(p, slug):
    taken = {h.get("slug") for h in churn.all_hyps(p)}
    if slug not in taken:
        return slug
    n = 2
    while f"{slug}-{n}" in taken:
        n += 1
    return f"{slug}-{n}"


def cmd_new(a):
    p = problem_of(a.problem)
    slug = re.sub(r"[^a-z0-9-]+", "-", a.slug.lower()).strip("-")[:60]
    if len(slug) < 3:
        print("slug too short", file=sys.stderr)
        sys.exit(2)
    slug = unique_slug(p, slug)
    expected = json.loads(a.expected) if a.expected else []
    if a.depends_on:
        dp, where = churn.find_hyp(p, a.depends_on)
        if not dp:
            print(f"depends_on {a.depends_on} not found in {p}", file=sys.stderr)
            sys.exit(2)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    fm = ["---", f"problem: {p}", f"slug: {slug}", f"created: {created}", "base: master",
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
           f"kind: {a.kind}", f"controls_for: {yaml_scalar(a.controls_for)}", "review: required", "status: queued", "---"]
    body = [f"# {slug}", "", "## Change", "", a.change.strip(), "", "## Expected", ""]
    body += [f"- `{e.get('shapes', '*')}` {e.get('direction', 'faster')}: {e.get('why', '')}" for e in expected] or ["- (none stated)"]
    body += ["", "## Risk", "", (a.notes or f"{a.risk} risk.").strip(), ""]
    qdir = os.path.join(REPO, p, "hypotheses", "queue")
    os.makedirs(qdir, exist_ok=True)
    path = os.path.join(qdir, f"{created}-{slug}.md")
    open(path, "w").write("\n".join(fm) + "\n\n" + "\n".join(body))
    errs, _ = churn.lint_hyp(path, p)
    if errs:
        os.remove(path)
        print(json.dumps({"ok": False, "errors": errs}))
        sys.exit(1)
    print(json.dumps({"ok": True, "file": os.path.relpath(path, REPO), "slug": slug, "problem": p}))


def cmd_list(a):
    probs = [problem_of(a.problem)] if a.problem else PROBLEMS
    for p in probs:
        for h in churn.all_hyps(p):
            s = h.get("speedup")
            print(f"{p} {h['_where']:16} {h.get('slug'):40} [{h.get('area')}] p{h.get('priority')} {h.get('outcome') or h.get('status')}" + (f" {s:.4f}x" if isinstance(s, (int, float)) else ""))


def cmd_lint(a):
    errs, h = churn.lint_hyp(a.file)
    print(json.dumps({"ok": not errs, "errors": errs, "slug": h.get("slug"), "problem": h.get("problem")}))
    sys.exit(1 if errs else 0)


def cmd_context(a):
    p = problem_of(a.problem)
    hd = os.path.join(REPO, p, "hypotheses")
    best = churn.load_best(p)
    print(json.dumps({
        "problem": p,
        "read_first": [f"{p}/hypotheses/FINDINGS.md", f"{p}/TASK.md", f"{p}/baseline.py", f"{p}/submission.py", "rules/technical-rules.llms.txt"],
        "best": best,
        "queue": [h.get("slug") for h in churn.all_hyps(p) if h["_where"] in ("queue", "running")],
        "done": [(h.get("slug"), h.get("outcome")) for h in churn.all_hyps(p) if h["_where"].startswith("done/")],
        "areas_hint": churn.AREAS_HINT,
        "submission_at_master": subprocess.run(["git", "rev-parse", "--short", f"HEAD:{p}/submission.py"], capture_output=True, text=True, cwd=REPO).stdout.strip(),
    }, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("new")
    x.add_argument("--problem", required=True); x.add_argument("--slug", required=True); x.add_argument("--area", required=True)
    x.add_argument("--change", required=True); x.add_argument("--expected"); x.add_argument("--risk", default="medium", choices=["low", "medium", "high"])
    x.add_argument("--depends-on"); x.add_argument("--priority", type=int, default=3); x.add_argument("--notes")
    x.add_argument("--kind", default="primary", choices=["primary", "ab-revert", "ab-tune", "control"],
                   help="primary = an idea of its own; ab-revert/ab-tune/control = an A/B companion queued by the adversarial review")
    x.add_argument("--controls-for", help="slug of the hypothesis this companion is an A/B control for")
    x.set_defaults(fn=cmd_new)
    x = sp.add_parser("list"); x.add_argument("problem", nargs="?"); x.set_defaults(fn=cmd_list)
    x = sp.add_parser("lint"); x.add_argument("file"); x.set_defaults(fn=cmd_lint)
    x = sp.add_parser("context"); x.add_argument("problem"); x.set_defaults(fn=cmd_context)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
