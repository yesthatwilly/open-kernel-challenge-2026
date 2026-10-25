#!/usr/bin/env python3
"""Token-cost ledger for churn agents, built from a Claude Code session transcript.

Each Agent spawn (description, model, prompt) is joined to the usage line of its completion
notification (<subagent_tokens>) and to its own transcript (<session>/subagents/agent-<id>.jsonl),
whose per-call usage gives the exact split into input, output, cache-read and cache-write tokens.
The hypothesis slug is the first known slug named in the prompt, the role comes from the spawn
description, and the outcome and speedup come from <p>/hypotheses/done/<outcome>/<slug>.result.json.
The main thread's own calls are summed per UTC day as orchestration overhead (not attributed).

Weighted tokens ("wtok") price each type at Anthropic API list-price ratios relative to plain input:
output 5x, cache write 1.25x, cache read 0.1x. They are a proxy for how much each run counts toward
usage limits, whose real weighting is not published.

Writes hypotheses/costs.jsonl (one row per agent run) and hypotheses/COSTS.md (predictions by role
and model, cost per outcome, per-hypothesis totals, main-thread overhead).

usage: costs.py <transcript.jsonl> [more transcripts...]
"""
import glob, json, os, re, statistics, sys
from collections import defaultdict

REPO = os.environ.get("CHURN_REPO", "/home/user/git/musespark")
PROBLEMS = ["muse_attention_decode", "muse_bf16_proj", "muse_decode_layer", "muse_moe_w4a16", "muse_prefill_layer"]
ROLES = [("review", r"^review"), ("build", r"^build"), ("audit", r"^audit"), ("mint", r"^mint"),
         ("read", r"(read|harvest)"), ("diagnose", r"^diagnose"), ("triage", r"^triage")]
TYPES = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]
WEIGHT = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_read_input_tokens": 0.1, "cache_creation_input_tokens": 1.25}


def known_slugs():
    slugs = {}
    for p in PROBLEMS:
        for f in glob.glob(os.path.join(REPO, p, "hypotheses", "**", "*.md"), recursive=True):
            m = re.match(r"^\d{4}-\d\d-\d\dT\d{6}Z-(.+)\.md$", os.path.basename(f))
            if m:
                slugs[m.group(1)] = p
    return slugs


def outcomes():
    out = {}
    for p in PROBLEMS:
        for f in glob.glob(os.path.join(REPO, p, "hypotheses", "done", "*", "*.result.json")):
            try:
                d = json.load(open(f))
            except (OSError, ValueError):
                continue
            out[(p, d.get("slug") or os.path.basename(f)[:-12])] = (os.path.basename(os.path.dirname(f)), d.get("speedup"))
    return out


def usage_of(events):
    """Sum per-type usage over assistant messages, counting each API message id once."""
    tot, seen, models = dict.fromkeys(TYPES, 0), set(), defaultdict(int)
    for ev in events:
        msg = ev.get("message") or {}
        u = msg.get("usage")
        if ev.get("type") != "assistant" or not u or msg.get("model") == "<synthetic>":
            continue
        if msg.get("id") in seen:
            continue
        seen.add(msg.get("id"))
        models[msg.get("model")] += 1
        for k in TYPES:
            tot[k] += u.get(k) or 0
    return tot, models


def wtok(u):
    return int(sum(WEIGHT[k] * u[k] for k in TYPES))


def read_jsonl(path):
    for line in open(path, errors="replace"):
        try:
            yield json.loads(line)
        except ValueError:
            continue


def sub_usage(agent, session_paths):
    for sp in session_paths:
        f = os.path.join(sp[:-len(".jsonl")], "subagents", f"agent-{agent}.jsonl")
        if os.path.exists(f):
            return usage_of(read_jsonl(f))
    return None, None


def main_overhead(session_paths):
    """Main-thread usage per UTC day across the given sessions."""
    days = defaultdict(list)
    for sp in session_paths:
        for ev in read_jsonl(sp):
            if ev.get("type") == "assistant" and not ev.get("isSidechain"):
                days[(ev.get("timestamp") or "")[:10]].append(ev)
    return {d: usage_of(evs)[0] for d, evs in sorted(days.items()) if d}


def role_of(desc):
    d = desc.lower()
    for r, pat in ROLES:
        if re.search(pat, d):
            return r
    return "other"


def parse(paths):
    spawns, tokens = {}, {}
    for path in paths:
        for line in open(path, errors="replace"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            msg = ev.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for c in content or []:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use" and c.get("name") == "Agent":
                    spawns[c["id"]] = {"desc": c["input"].get("description", ""),
                                       "model": c["input"].get("model") or "inherit",
                                       "prompt": c["input"].get("prompt", ""), "ts": ev.get("timestamp")}
                elif c.get("type") == "tool_result" and c.get("tool_use_id") in spawns:
                    txt = c.get("content")
                    txt = json.dumps(txt) if not isinstance(txt, str) else txt
                    m = re.search(r"agentId: (a[0-9a-f]+)", txt)
                    if m:
                        spawns[c["tool_use_id"]]["agent"] = m.group(1)
            blob = line if "subagent_tokens" in line else ""
            for m in re.finditer(r"<task-id>(a[0-9a-f]+)</task-id>.*?<subagent_tokens>(\d+)</subagent_tokens>"
                                 r"<tool_uses>(\d+)</tool_uses><duration_ms>(\d+)</duration_ms>", blob):
                tid, tok, tu, ms = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
                prev = tokens.get(tid)
                # a resumed agent notifies again; keep the largest (latest cumulative) figure
                if not prev or tok > prev[0]:
                    tokens[tid] = (tok, tu, ms)
    return spawns, tokens


def main():
    paths = sys.argv[1:]
    slugs = known_slugs()
    by_len = sorted(slugs, key=len, reverse=True)
    spawns, tokens = parse(paths)
    outs = outcomes()
    rows = []
    for s in spawns.values():
        a = s.get("agent")
        if not a or a not in tokens:
            continue
        tok, tu, ms = tokens[a]
        hit = [x for x in by_len if x in s["prompt"]]
        slug = hit[0] if len(hit) >= 1 else None
        u, models = sub_usage(a, paths)
        oc, sp = outs.get((slugs.get(slug), slug), (None, None))
        rows.append({"agent": a, "ts": s["ts"], "role": role_of(s["desc"]), "model": s["model"],
                     "models": dict(models) if models else None,
                     "desc": s["desc"], "slug": slug, "problem": slugs.get(slug), "outcome": oc, "speedup": sp,
                     "tokens": tok, "usage": u, "wtok": wtok(u) if u else None,
                     "tool_uses": tu, "minutes": round(ms / 60000, 1),
                     "multi": len(set(hit)) > 1 and s["desc"].lower().startswith(("review six", "review five", "mint"))})
    rows.sort(key=lambda r: r["ts"] or "")
    out = os.path.join(REPO, "hypotheses")
    with open(os.path.join(out, "costs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    per_hyp = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["slug"] and not r["multi"] and r["role"] in ("review", "build", "audit"):
            per_hyp[(r["problem"], r["slug"])][r["role"]] += r["tokens"]
            per_hyp[(r["problem"], r["slug"])]["wtok"] += r["wtok"] or 0
    role_tok, role_w = defaultdict(list), defaultdict(list)
    for r in rows:
        if not r["multi"]:
            role_tok[(r["role"], r["model"])].append(r["tokens"])
            if r["wtok"] is not None:
                role_w[(r["role"], r["model"])].append(r["wtok"])
    steps = {"review", "build", "audit"}
    full = [sum(v[k] for k in steps) for v in per_hyp.values() if steps <= set(v)]
    covered = [r for r in rows if r["usage"]]
    sub_u = {k: sum(r["usage"][k] for r in covered) for k in TYPES}
    main_u = main_overhead(paths)

    def fmt_u(u):
        return " | ".join(f"{u[k]:,}" for k in TYPES) + f" | {wtok(u):,}"

    L = ["# Agent token costs", "",
         "Generated by `.claude/skills/churnsubs/bin/costs.py` from the session transcript(s); raw rows in",
         "`costs.jsonl`. \"tokens\" is each subagent's total from its completion notification. \"wtok\" weights",
         "the exact per-type usage from the subagent's own transcript at API list-price ratios (output 5x,",
         "cache write 1.25x, cache read 0.1x, input 1x): a proxy for usage-limit cost, not the real weighting.", "",
         f"Agents counted: {len(rows)}; total subagent tokens: {sum(r['tokens'] for r in rows):,}; "
         f"{len(covered)} have a transcript with the per-type split.", "",
         "## Usage by type", "", "| scope | input | output | cache read | cache write | wtok |", "|---|---|---|---|---|---|",
         f"| subagents (with transcript) | {fmt_u(sub_u)} |",
         f"| main thread (all days) | {fmt_u({k: sum(u[k] for u in main_u.values()) for k in TYPES})} |", "",
         "## Prediction table (per agent run)", "",
         "| role | model | runs | median tokens | p90 tokens | median wtok | p90 wtok |", "|---|---|---|---|---|---|---|"]
    for (role, model), v in sorted(role_tok.items()):
        v, w = sorted(v), sorted(role_w.get((role, model), []))
        wm = f"{int(statistics.median(w)):,} | {w[int(0.9 * (len(w) - 1))]:,}" if w else "- | -"
        L.append(f"| {role} | {model} | {len(v)} | {int(statistics.median(v)):,} | {v[int(0.9 * (len(v) - 1))]:,} | {wm} |")
    if full:
        L += ["", f"One hypothesis through review + build + audit: median {int(statistics.median(full)):,} tokens, "
              f"mean {int(statistics.mean(full)):,} over {len(full)} hypotheses (excludes batched reviews and the main thread)."]
    by_oc = defaultdict(list)
    for (p, sl), v in per_hyp.items():
        by_oc[outs.get((p, sl), ("open", None))[0]].append(v)
    gains = len(by_oc.get("gain", []))
    all_w = sum(v["wtok"] for v in per_hyp.values())
    L += ["", "## Cost by outcome", "", "Per-hypothesis review + build + audit spend, grouped by how the run ended.", "",
          "| outcome | hypotheses | median tokens | total tokens | total wtok |", "|---|---|---|---|---|"]
    for oc, vs in sorted(by_oc.items(), key=lambda kv: -len(kv[1])):
        t = [sum(v[k] for k in steps) for v in vs]
        L.append(f"| {oc} | {len(vs)} | {int(statistics.median(t)):,} | {sum(t):,} | {sum(v['wtok'] for v in vs):,} |")
    if gains:
        L += ["", f"Spend per gain: {sum(sum(v[k] for k in steps) for v in per_hyp.values()) // gains:,} tokens, "
              f"{all_w // gains:,} wtok ({gains} gains over {len(per_hyp)} attributed hypotheses)."]
    L += ["", "## Main-thread overhead per UTC day", "", "Orchestration in the main conversation; not attributed to hypotheses.", "",
          "| day | input | output | cache read | cache write | wtok |", "|---|---|---|---|---|---|"]
    for d, u in main_u.items():
        L.append(f"| {d} | {fmt_u(u)} |")
    L += ["", "## Per hypothesis", "", "| problem | slug | outcome | speedup | review | build | audit | total | wtok |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (p, sl), v in sorted(per_hyp.items(), key=lambda kv: -sum(kv[1][k] for k in steps)):
        oc, sp = outs.get((p, sl), ("open", None))
        L.append(f"| {p.replace('muse_', '')} | {sl} | {oc} | {f'{sp:.4f}' if sp else '-'} | {v.get('review', 0):,} | "
                 f"{v.get('build', 0):,} | {v.get('audit', 0):,} | {sum(v[k] for k in steps):,} | {v['wtok']:,} |")
    open(os.path.join(out, "COSTS.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps({"rows": len(rows), "hypotheses": len(per_hyp), "total": sum(r["tokens"] for r in rows)}))


if __name__ == "__main__":
    main()
