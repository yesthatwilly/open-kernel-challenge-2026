"""Fill a churnsubs prompt template for one hypothesis and print the output path.

usage: fill.py adversary|audit <problem> <slug> [implementer]

Filled prompts go to <repo>/hypotheses/prompts/filled/ (git-ignored by its own .gitignore; the node's /tmp is a small,
nearly full tmpfs), or $FILL_OUT.
"""
import glob, json, os, subprocess, sys

REPO = "/home/user/git/musespark"
SKILL = f"{REPO}/.claude/skills/churnsubs"
os.environ.setdefault("CHURN_REPO", REPO)
sys.path.insert(0, f"{REPO}/.claude/skills/hypothesize/bin")
import hyp  # noqa: E402  (claims, the active campaign, churn's best / margin / frontmatter)

churn = hyp.churn
kind, p, slug = sys.argv[1:4]
p = churn.problem_of(p)
impl = sys.argv[4] if len(sys.argv) > 4 else "a Sonnet subagent"
OUT = os.environ.get('FILL_OUT') or hyp.scratch_dir()


def hyp_path():
    for d in ("queue", "running"):
        m = glob.glob(f"{REPO}/{p}/hypotheses/{d}/*-{slug}.md")
        if m:
            return os.path.relpath(m[0], REPO)
    sys.exit(f"no hypothesis {p}/{slug}")


def freshness(h):
    """One line: the board and blob the file was unheld / reviewed / minted on (churn's _fresh_ref, the gate's own
    reference) against today's BEST and master, and the gate's verdict."""
    best = churn.load_best(p) or {}
    now_b, blob = best.get("speedup"), churn._master_blob(p)
    what, at, then, seen = churn._fresh_ref(p, h)
    moved = abs(now_b / then - 1) if then and now_b else None
    g = churn.freshness(p, h, best, blob)
    return (f"{what} {at} on board {f'{then:.4f}' if then else 'unknown'}" + (f", blob {str(seen)[:8]}" if seen else "")
            + f"; BEST now {now_b:.4f} (job {best.get('job')}), master blob {blob[:8]}"
            + (f"; moved {moved * 100:.2f}% against a {churn.margin() * 100:.2f}% margin" if moved is not None else "")
            + (f"; freshness gate: {g[0]} (re-derive the value case on today's rows)" if g else "; freshness gate: fresh")
            + f"; value case: {'EV ' + str(h.get('value_ev')) + '%' if h.get('value_ev') is not None else 'none (minted before the value bar)'}.")


md = hyp_path()
if kind == "adversary":
    h = churn.read_hyp(os.path.join(REPO, md))
    landed = subprocess.run(["python3", "-I", f"{SKILL}/bin/churn.py", "landed", p], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    others = sorted({os.path.basename(f)[20:-3] for d in ("queue", "running")
                     for f in glob.glob(f"{REPO}/{p}/hypotheses/{d}/*.md")} - {slug})
    camp = hyp.active_campaign()
    cfiles = [f for f in (f"{camp}/FINDINGS.md", f"{camp}/DISTILL.md") if camp and os.path.exists(f"{REPO}/{f}")]
    hits = hyp.matching_claims(p, slug, h.get("area"), h.get("change"), hyp.CLAIM_TAGS, h.get("expected"))
    t = open(f"{SKILL}/ADVERSARY_PROMPT.md").read()
    for k, v in {"REPO": REPO, "HYP_MD": md, "FINDINGS": f"{p}/hypotheses/FINDINGS.md",
                 "CLAIMS": f"{p}/hypotheses/CLAIMS.md" if os.path.exists(hyp.claims_path(p)) else "(no CLAIMS.md yet for this problem)",
                 "CLAIM_HITS": "\n".join(f"   - {p}/hypotheses/CLAIMS.md:{c['line']} ({', '.join(w)}): {c['text']}" for c, w in hits)
                               or "   - none name this slug or share its distinctive words",
                 "CAMPAIGN": (", ".join(cfiles) + f", and the top {hyp.LOG_HEAD} lines of {camp}/LOG.md (the Now block)")
                             if camp else "(no active campaign)",
                 "FRESHNESS": freshness(h),
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
