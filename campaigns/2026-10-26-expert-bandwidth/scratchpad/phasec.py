#!/usr/bin/env python3
"""Stage and record phase-C variant checks (campaign 2026-10-26-expert-bandwidth, 3:1 rhythm).

A variant generator leaves, on a branch in a worktree, one commit per variant plus
campaigns/2026-10-26-expert-bandwidth/phasec/<tag>/MANIFEST.jsonl and an attempt draft <slug>.md.

usage:
  phasec.py list    <worktree> <tag>                 manifest rows with their dispatch state
  phasec.py prep    <worktree> <tag> <problem> <slug> write the variant's submission.py, verify sha256,
                                                      copy the draft to <problem>/attempts/<UTC>-check-<slug>.md,
                                                      print the bare okc command to run next
  phasec.py record  <worktree> <tag> <problem> <slug> <job>
                                                      fill the Job line, commit the attempt file

okc is never run from here: the main thread runs the printed command as a bare Bash call so the
repo's attempt gate (.claude/hooks/okc-attempt-gate.py) fires.
"""
import datetime, glob, hashlib, json, os, re, subprocess, sys

CAMP = "campaigns/2026-10-26-expert-bandwidth"


def git(wt, *a):
    return subprocess.check_output(["git", "-C", wt, *a], text=True)


def manifest(wt, tag):
    p = os.path.join(wt, CAMP, "phasec", tag, "MANIFEST.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


def row(wt, tag, slug):
    for r in manifest(wt, tag):
        if r["slug"] == slug:
            return r
    sys.exit(f"no variant {slug} in {tag}")


def dispatched(wt, problem, slug):
    return sorted(glob.glob(os.path.join(wt, problem, "attempts", f"*-check-{slug}.md")))


def cmd_list(wt, tag):
    for r in manifest(wt, tag):
        prob = r.get("problem") or ""
        done = [os.path.basename(f) for f in glob.glob(os.path.join(wt, "muse_*", "attempts", f"*-check-{r['slug']}.md"))]
        print(f"{r['slug']:<44} {r['commit'][:8]} {r['bytes']:>8} {'DISPATCHED ' + done[-1][:16] if done else 'ready'}")


def cmd_prep(wt, tag, problem, slug):
    r = row(wt, tag, slug)
    if dispatched(wt, problem, slug):
        sys.exit(f"{slug} already has an attempt file: {dispatched(wt, problem, slug)[-1]}")
    data = subprocess.check_output(["git", "-C", wt, "show", f"{r['commit']}:{problem}/submission.py"])
    sha = hashlib.sha256(data).hexdigest()
    if sha != r["sha256"]:
        sys.exit(f"sha256 mismatch for {slug}: commit has {sha[:12]}, manifest {r['sha256'][:12]}")
    if not data.startswith(f"# okc: problem={problem}".encode()):
        sys.exit("first line is not the okc problem marker")
    sub = os.path.join(wt, problem, "submission.py")
    open(sub, "wb").write(data)
    draft = open(os.path.join(wt, CAMP, "phasec", tag, f"{slug}.md")).read()
    if not re.search(r"^- \*\*Job:\*\* `<job id>`\s*$", draft, re.M):
        sys.exit("draft lacks the exact Job placeholder line")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
    att = os.path.join(wt, problem, "attempts", f"{ts}-check-{slug}.md")
    open(att, "w").write(draft.replace("(dispatch time)", ts))
    print(json.dumps({"slug": slug, "sha256": sha[:16], "bytes": len(data), "attempt": att,
                      "run": f"okc check {problem} {os.path.abspath(sub)} --no-wait"}))


def cmd_record(wt, tag, problem, slug, job):
    att = dispatched(wt, problem, slug)
    if not att:
        sys.exit("no attempt file to record into")
    att = att[-1]
    s = open(att).read()
    ts = os.path.basename(att)[:16]
    s2, n = re.subn(r"^- \*\*Job:\*\* `<job id>`\s*$", f"- **Job:** `{job}` (`okc status {job}`), dispatched {ts}", s, flags=re.M)
    if n != 1:
        sys.exit("Job placeholder not found exactly once")
    open(att, "w").write(s2)
    rel = os.path.relpath(att, wt)
    git(wt, "add", rel)
    git(wt, "commit", "-q", "-m", f"attempts: phasec {tag} {slug} check {job} (pending)\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    print(f"recorded {slug} -> {job} in {rel}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    {"list": cmd_list, "prep": cmd_prep, "record": cmd_record}[a[0]](*a[1:])
