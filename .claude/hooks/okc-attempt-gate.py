#!/usr/bin/env python3
"""PreToolUse gate for `okc check` / `okc submit`.

Refuses the run unless the problem's attempts/ directory holds an unconsumed attempt file for
the same mode: named <YYYY-MM-DDTHHMM[SS]Z>-<mode>-<slug>.md, modified within the last 2 hours, and
whose Job line is still the template placeholder (no job id filled in yet). Every run needs its
own file, so a file whose job id has been filled in does not cover a second run.

The attempts directory is the one in the checkout that owns the submission file, so a worktree's
submission is gated by the worktree's own attempts/, not the main tree's.

Reads the hook JSON on stdin, writes a permission decision as JSON on stdout. Anything that is
not an okc check/submit command passes through untouched.
"""
import json
import os
import re
import shlex
import sys
import time

MAX_AGE_S = 2 * 60 * 60
PLACEHOLDER = re.compile(r"\*\*Job:\*\*\s*`?<job id>`?")
FIRST_LINE = re.compile(r"^#\s*okc:\s*problem=(\S+)")


def allow():
    sys.exit(0)


def deny(reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def repo_root(start):
    """Walk up to the nearest git checkout: a `.git` directory (main tree) or `.git` file (worktree)."""
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return None


def parse_invocations(command):
    """Yield (mode, positionals) for each `okc check|submit` in the command line."""
    try:
        toks = shlex.split(command)
    except ValueError:
        toks = command.split()
    i = 0
    while i < len(toks):
        if os.path.basename(toks[i]) == "okc" and i + 1 < len(toks) and toks[i + 1] in ("check", "submit"):
            mode = toks[i + 1]
            pos = []
            j = i + 2
            while j < len(toks) and toks[j] not in ("&&", "||", ";", "|"):
                if not toks[j].startswith("-"):
                    pos.append(toks[j])
                j += 1
            yield mode, pos
            i = j
        else:
            i += 1


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        allow()
    if payload.get("tool_name") != "Bash":
        allow()
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bokc\s+(check|submit)\b", command):
        allow()
    cwd = payload.get("cwd") or os.getcwd()

    for mode, pos in parse_invocations(command):
        if not pos:
            deny(f"okc {mode}: no file given; the gate needs the submission path to find its problem.")
        if len(pos) >= 2:
            problem, path = pos[0], pos[1]
        else:
            problem, path = None, pos[0]
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)
        if not os.path.isfile(path):
            deny(f"okc {mode}: {path} not found from cwd {cwd}; give a path resolvable from the session cwd.")
        if problem is None:
            with open(path, errors="replace") as fh:
                m = FIRST_LINE.match(fh.readline())
            if not m:
                deny(f"okc {mode}: {path} has no '# okc: problem=' first line, so the gate cannot find its problem.")
            problem = m.group(1)
        # The checkout that owns the submission file (main tree or a worktree) holds its attempts.
        root = repo_root(os.path.dirname(path)) or cwd
        attempts = os.path.join(root, problem, "attempts")
        if not os.path.isdir(attempts):
            deny(f"okc {mode} refused: {attempts} does not exist. Write the attempt file first "
                 f"(see ATTEMPT_TEMPLATE.md at the repo root).")
        now = time.time()
        fresh, stale, consumed = [], [], []
        for name in sorted(os.listdir(attempts)):
            if not re.match(r"\d{4}-\d{2}-\d{2}T\d{4}(\d{2})?Z-" + mode + r"-.+\.md$", name):
                continue
            full = os.path.join(attempts, name)
            age = now - os.path.getmtime(full)
            with open(full, errors="replace") as fh:
                text = fh.read()
            if not PLACEHOLDER.search(text):
                consumed.append(name)
            elif age > MAX_AGE_S:
                stale.append(name)
            else:
                fresh.append(name)
        if fresh:
            allow()
        hint = (f"okc {mode} refused for {problem}: no unconsumed attempt file in {attempts}. "
                f"Every run needs its own file named <YYYY-MM-DDTHHMM[SS]Z>-{mode}-<slug>.md, modified in the "
                f"last 2 hours, with its Job line still '`<job id>`' (fill the id in after the run). "
                f"Copy ATTEMPT_TEMPLATE.md from the repo root.")
        if stale:
            hint += f" Stale (older than 2 h): {', '.join(stale[-3:])}."
        if consumed:
            hint += f" Already consumed: {', '.join(consumed[-3:])}."
        deny(hint)
    allow()


if __name__ == "__main__":
    main()
