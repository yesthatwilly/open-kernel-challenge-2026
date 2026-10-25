"""Fill a churnsubs prompt template for one hypothesis and print the output path.

usage: fill.py adversary|audit <problem> <slug> [implementer]
"""
import glob, json, os, subprocess, sys

REPO = "/home/user/git/musespark"
SKILL = f"{REPO}/.claude/skills/churnsubs"
OUT = os.environ.get('FILL_OUT', '/tmp/churn-prompts')
kind, p, slug = sys.argv[1:4]
impl = sys.argv[4] if len(sys.argv) > 4 else "a Sonnet subagent"


def hyp_path():
    for d in ("queue", "running"):
        m = glob.glob(f"{REPO}/{p}/hypotheses/{d}/*-{slug}.md")
        if m:
            return os.path.relpath(m[0], REPO)
    sys.exit(f"no hypothesis {p}/{slug}")


md = hyp_path()
if kind == "adversary":
    landed = subprocess.run(["python3", "-I", f"{SKILL}/bin/churn.py", "landed", p], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    others = sorted({os.path.basename(f)[20:-3] for d in ("queue", "running")
                     for f in glob.glob(f"{REPO}/{p}/hypotheses/{d}/*.md")} - {slug})
    t = open(f"{SKILL}/ADVERSARY_PROMPT.md").read()
    for k, v in {"REPO": REPO, "HYP_MD": md, "FINDINGS": f"{p}/hypotheses/FINDINGS.md",
                 "BEST_JSON": f"{p}/hypotheses/BEST.json", "LANDED_JSON": landed,
                 "BASE_SUBMISSION": f"{p}/submission.py (master)", "PROBLEM": p,
                 "OTHER_SLUGS": ", ".join(others)}.items():
        t = t.replace("{" + k + "}", v)
else:
    t = open(f"{SKILL}/AUDIT_PROMPT.md").read()
    for k, v in {"PROBLEM": p, "HYP_MD": md, "WORKTREE": f"{REPO}/.worktrees/{p}/{slug}",
                 "IMPLEMENTER": impl}.items():
        t = t.replace("{" + k + "}", v)
os.makedirs(OUT, exist_ok=True)
path = f"{OUT}/{kind}-{p}-{slug}.md"
open(path, "w").write(t)
print(path)
