#!/usr/bin/env python3
"""Build data/times.json: per-landing, per-shape kernel timelines of muse_decode_layer.

Reproducible, stdlib only, read-only on the repo (git show / git log, profile files).
Writes only data/times.json next to this script.

  python3 campaigns/2026-11-05-kernel-viz/collect_times.py            # write data/times.json + validation
  python3 campaigns/2026-11-05-kernel-viz/collect_times.py --names    # dump distinct kernel names per era

Sources per landing (first hit wins):
  profiles/<job>.json
  muse_decode_layer/profiles/submission.<job>.json
  muse_decode_layer/hypotheses/archive/*/<job>.profile.json.gz
  campaigns/*/runs/<job>*.profile.json.gz          (also inside .worktrees/*)
  .worktrees/**/profiles/*.<job>.json
  campaigns/*/runs/<job>.json.gz  (okc status file: result.records[event=profile|benchmark])
Kernel -> stage mapping: stage_map.tsv (era list, regex, stage, note); the regex is searched in
'<name>@calls=<n>' and the first row whose era list holds the landing's era wins; no match ->
unattributed.
"""
import glob, gzip, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
STAGES = ["norm", "qkv", "attention", "o_proj", "router", "experts", "combine", "unattributed"]

# Master landings of muse_decode_layer/submission.py (first-parent master), chronological.
# (commit, job, job_match, era, note).  Non-landing commits that touched submission.py are
# excluded: ab21d709 (starter kit), b34cde34/4c0d905c/3676829f/3549aee6 (10-30 swap-AB
# containment, revert, disable, restore-to-scored-bytes) and e23ecdb5 (restore of eb78a8c0's
# scored bytes; folded into eb78a8c0).
LANDINGS = [
    ("164c1d27", "e260dfe413e448c7", "confirmed", "sglang", "job from commit message (BEST.json at this commit names 5755a412, 1.02x)"),
    ("ef7dea6c", "0730135cdcac43b7", "confirmed", "ours", ""),
    ("09da1e0d", "fb07a3bd3aa54baf", "confirmed", "ours", ""),
    ("ea896254", "9dcaec316b3045a7", "confirmed", "ours", ""),
    ("f1f2f35c", "81ee6714b7614a99", "confirmed", "ours", ""),
    ("8fa8fbf0", "cc92097b64784c0f", "confirmed", "ours", ""),
    ("b047510f", "c243a29f26dc4725", "confirmed", "ours", ""),
    ("00e36e3a", "7a9f5de5aecc4aba", "confirmed", "ours", ""),
    ("d5595769", "34a9be06918d448b", "confirmed", "ours", ""),
    ("6773ec5c", "933d3f825a044d36", "confirmed", "ours", ""),
    ("ec89382b", "a903b9d004d2452d", "confirmed", "ours", ""),
    ("ce625b00", "69db8aeb251944c5", "by_speedup", "ours", "commit names no job; decode-first LOG.md 05:30Z names scored 69db8aeb at 1.3408"),
    ("ec719761", "c3b7440aa9394c04", "by_speedup", "ours", "commit names no job; decode-first LOG.md 06:32Z names scored run c3b7440a"),
    ("c8b77a59", "efa0e184b7914d80", "by_speedup", "ours", "commit names no job; decode-first LOG.md 07:45Z names scored efa0e184 at 1.3840"),
    ("249f6ef3", "f8accdd5c5cc4923", "by_speedup", "ours", "commit names no job; decode-first LOG.md 07:38Z names scored f8accdd5 at 1.3928"),
    ("2fc609f3", "b96335161c494174", "by_speedup", "ours", "commit names no job; decode-first LOG.md 08:05Z names scored b9633516 at 1.4223"),
    ("2196b906", "0d84f58ca33b424b", "by_speedup", "ours", "commit names no job; decode-first LOG.md/RETRO.md name scored 0d84f58c at 1.4422"),
    ("3cd54072", "a03e596d3aaa4f69", "by_speedup", "ours", "commit names no job; expert-bandwidth LOG.md 15:51Z names scored a03e596d at 1.4466"),
    ("49142841", "dc51d8e8f58e4cb7", "by_speedup", "ours", "commit names no job; expert-bandwidth LOG.md 17:05Z names scored dc51d8e8 at 1.4956"),
    ("4adf7b75", "2bd373c9e23c4667", "confirmed", "ours", ""),
    ("76815f82", "42b3e39478474bcd", "confirmed", "ours", ""),
    ("07b71687", "ba57a64f66864219", "confirmed", "ours", "profile only in the okc status file"),
    ("d5eeef58", "7ceab934eea24099", "confirmed", "ours", "profile only in the okc status file"),
    ("dca993b1", "4a7a49d0ae82440f", "confirmed", "ours", ""),
    ("3e176f6c", "c1daf58bf192471e", "confirmed", "ours", ""),
    ("d6c1a9b1", "e1fd3c00c83045dc", "confirmed", "ours", ""),
    ("622b97bb", "adb9c3436aad4e1d", "confirmed", "ours", ""),
    ("f3828162", "3a68058a0f0a4eab", "confirmed", "ours", ""),
    ("f8cf2476", "9afcccc328974350", "confirmed", "ours", ""),
    ("d0cdde04", "7352add6920f40e7", "confirmed", "ours", ""),
    ("ec3c4dcb", "83f0c00eb3644e1c", "confirmed", "ours", ""),
    ("d1fb10ab", "31e9a69aac344675", "confirmed", "ours", ""),
    ("d28b23d7", "7cc034f917274439", "confirmed", "ours", ""),
    ("10442ca2", "824b60fac87c4ff4", "confirmed", "ours", ""),
    ("787c0eac", "9263781bc2a040e9", "confirmed", "ours", ""),
    ("594adec8", "7c7524a61490426f", "confirmed", "ours", "no judge profile found"),
    ("5dfc6d96", "9299f06d2d854864", "confirmed", "ours", "no judge profile found"),
    ("eb78a8c0", "44de510326bb4e02", "confirmed", "ours", "no judge profile found; e23ecdb5 (07:46Z) restored these scored bytes after eb78a8c0 took an unscored tip"),
    ("70446a48", "43265863258a47a0", "confirmed", "ours", "no judge profile found"),
    ("d07539ae", "dfb318ee06b842f7", "confirmed", "ours", ""),
    ("bf25b827", "63d0785130904bd1", "confirmed", "ours", "no judge profile found; follows the 10-30 swap-AB containment commits"),
    ("36584c5c", "201db0b8c295487b", "confirmed", "public", "public-C D0 base"),
    ("991a2787", "9d726d84649e4f9e", "confirmed", "public", ""),
    ("ce08a099", "545b80bb25904f3e", "confirmed", "public", ""),
    ("5af7e6a8", "9185f31ce229462b", "confirmed", "public", ""),
    ("6eefb7a7", "c3eadc8813204a58", "confirmed", "public", ""),
    ("897c618e", "85e5e041879b454a", "confirmed", "public", "profile only in .worktrees/trx-study"),
    ("c3b07763", "2d48f00fca8e40da", "confirmed", "keell", "keell0renz decode file adopted verbatim; profile only in .worktrees/trx-study"),
    ("c3ec85f7", "a02997bf3e0542ce", "confirmed", "keell", "profile only in .worktrees/dec-zl2"),
]


def git(*a):
    env = dict(os.environ, TZ="UTC")  # format-local dates in UTC
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True, env=env).stdout


def commit_date(c):
    return git("show", "-s", "--format=%cd", "--date=format-local:%Y-%m-%dT%H:%MZ", c).strip()


def load_json(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_profile(job):
    """Return (path, benchmarks, profiles, speedup) or None."""
    cands = [os.path.join(REPO, "profiles", job + ".json"),
             os.path.join(REPO, "muse_decode_layer", "profiles", "submission.%s.json" % job)]
    cands += sorted(glob.glob(os.path.join(REPO, "campaigns", "*", "runs", job + "*.profile.json.gz")))
    cands += sorted(glob.glob(os.path.join(REPO, ".worktrees", "*", "campaigns", "*", "runs", job + "*.profile.json.gz")))
    cands += sorted(glob.glob(os.path.join(REPO, "muse_decode_layer", "hypotheses", "archive", "*", job + ".profile.json.gz")))
    cands += sorted(glob.glob(os.path.join(REPO, ".worktrees", "**", "profiles", "*." + job + ".json"), recursive=True))
    for p in cands:
        if os.path.exists(p):
            try:
                d = load_json(p)
                b, pr = d.get("benchmarks") or [], d.get("profiles") or []
                if len(pr) == 12:
                    return p, b, pr, (d.get("summary") or {}).get("speedup")
            except Exception:
                pass
    st = sorted(glob.glob(os.path.join(REPO, "campaigns", "*", "runs", job + ".json.gz")))
    st += sorted(glob.glob(os.path.join(REPO, ".worktrees", "*", "campaigns", "*", "runs", job + ".json.gz")))
    for p in st:
        try:
            r = load_json(p).get("result") or {}
        except Exception:
            continue
        recs = r.get("records") or []
        pr = [x for x in recs if x.get("event") == "profile"]
        b = [x for x in recs if x.get("event") == "benchmark"]
        if len(pr) == 12:
            return p, b, pr, r.get("speedup")
    return None


ROW = re.compile(r"^(.*?)\s+([0-9.]+) µs\s+([0-9.]+)%\s+(\d+)\s*$")
TOT = re.compile(r"^total\s+([0-9.]+) µs")
BUSY = re.compile(r"^GPU busy.*?\s([0-9.]+) µs")


def parse_report(txt):
    rows, total, busy = [], None, None
    for ln in txt.splitlines():
        s = ln.rstrip()
        if not s or s.startswith("kernel ") or s.startswith("... ") or s.startswith("an overlapping"):
            continue
        m = TOT.match(s)
        if m:
            total = float(m.group(1)); continue
        m = BUSY.match(s)
        if m:
            busy = float(m.group(1)); continue
        m = ROW.match(s)
        if m:
            rows.append((m.group(1).strip(), float(m.group(2)), int(m.group(4))))
    return rows, total, busy


def base_name(name):
    s = name
    for pre in ("void ",):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r"^\(anonymous namespace\)::", "", s)
    s = s.split("<", 1)[0].split("(", 1)[0]
    s = s.replace("...", "")
    s = s.split("::")[-1] if "::" in s else s
    s = re.sub(r"(sk2_kernel_gle)[0-9a-f]+$", r"\1", s)
    s = re.sub(r"(sk2_kernel_narrow_)[0-9a-f]+$", r"sk2_kernel_narrow", s)
    return s.strip() or name


def case_shape(c):
    if isinstance(c, str):
        import ast
        c = ast.literal_eval(c)
    load = c["load"].split()[0]
    return {"load": load, "batch": int(c["batch"]), "layer": c["layer"], "q_len": int(c.get("q_len", 1)),
            "label": "%s b%d %s" % (load, int(c["batch"]), c["layer"])}


def load_stage_map():
    rows = []
    with open(os.path.join(HERE, "stage_map.tsv"), encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i == 0 or not ln.strip() or ln.startswith("#"):
                continue
            era, rx, stage, note = (ln.rstrip("\n").split("\t") + [""] * 4)[:4]
            assert stage in STAGES, (stage, ln)
            rows.append((set(e.strip() for e in era.split(",")), re.compile(rx), stage))
    return rows


def stage_of(smap, era, name, calls):
    key = "%s@calls=%d" % (name, calls)
    for eras, rx, st in smap:
        if era in eras and rx.search(key):
            return st
    return "unattributed"


def main():
    names_mode = "--names" in sys.argv
    smap = None if names_mode else load_stage_map()
    out = {"shapes": [], "landings": [], "frames": {}, "stage_order": STAGES}
    missing, names = [], {}
    final_bench = None
    for i, (c, job, match, era, note) in enumerate(LANDINGS):
        full = git("rev-parse", c).strip()
        prof = find_profile(job)
        L = {"i": i, "commit": c, "date": commit_date(c), "job": job, "speedup": None, "era": era,
             "has_profile": prof is not None, "job_match": match, "note": note}
        if prof is None:
            missing.append((c, job))
            d = load_best_speedup(full, job)
            if d is not None:
                L["speedup"] = round(d, 4)
        else:
            p, bench, profs, sp = prof
            L["speedup"] = round(sp, 4) if sp is not None else None
            if bench and len(bench) == 12:
                L["sglang_us"] = [round(b["baseline_ns"] / 1000.0, 2) for b in bench]
            if c == LANDINGS[-1][0]:
                final_bench = bench
            frames = []
            for sp_i, pr in enumerate(profs):
                rows, total, busy = parse_report(pr["report"])
                ks = []
                for nm, us, calls in rows:
                    b = base_name(nm)
                    names.setdefault(era, {}).setdefault(nm, [0, 0.0, c])
                    names[era][nm][0] += 1; names[era][nm][1] += us
                    st = "unattributed" if names_mode else stage_of(smap, era, nm, calls)
                    ks.append({"name": nm, "base": b, "stage": st, "us": us, "calls": calls})
                vis = sum(k["us"] for k in ks)
                other = round(max(0.0, total - vis), 2) if total is not None else 0.0
                if total is None:
                    total = round(vis, 2)
                frames.append({"total_us": total, "busy_us": busy, "other_us": other, "kernels": ks})
            out["frames"][c] = frames
        out["landings"].append(L)
    if final_bench is None:
        sys.exit("final landing has no profile")
    for j, b in enumerate(final_bench):
        s = case_shape(b["case"])
        out["shapes"].append({"i": j, "label": s["label"], "load": s["load"], "batch": s["batch"],
                              "layer": s["layer"], "q_len": s["q_len"],
                              "sglang_us": round(b["baseline_ns"] / 1000.0, 2)})
    if names_mode:
        for era, d in names.items():
            print("==", era)
            for nm, (n, us, c) in sorted(d.items(), key=lambda x: -x[1][1]):
                print("%7.1f %4d %s  %s" % (us, n, c, nm))
        return
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "times.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    validate(out, missing)


def load_best_speedup(full, job):
    """Speedup that BEST.json records for this job, at the landing commit or a later commit touching BEST.json."""
    later = git("rev-list", "--reverse", "--first-parent", full + "..master", "--",
                "muse_decode_layer/hypotheses/BEST.json").split()
    for c in [full] + later[:6]:
        try:
            d = json.loads(git("show", c + ":muse_decode_layer/hypotheses/BEST.json"))
        except Exception:
            continue
        if d.get("job") == job:
            return float(d["speedup"])
    return None


def validate(out, missing):
    bad = 0
    neg = 0
    for c, fr in out["frames"].items():
        for j, f in enumerate(fr):
            s = sum(k["us"] for k in f["kernels"]) + f["other_us"]
            if abs(s - f["total_us"]) > 0.5:
                bad += 1
                print("SUM MISMATCH", c, j, round(s, 2), f["total_us"])
    nfr = sum(len(v) for v in out["frames"].values())
    print("frames: %d landings x 12 = %d shape-frames; sum check failures: %d" % (len(out["frames"]), nfr, bad))
    era_of = {L["commit"]: L["era"] for L in out["landings"]}
    agg = {}
    for c, fr in out["frames"].items():
        a = agg.setdefault(era_of[c], {"tot": 0.0, "vis": 0.0, "un": 0.0, "names": set(), "rows": 0, "unrows": 0})
        for f in fr:
            a["tot"] += f["total_us"]
            for k in f["kernels"]:
                a["vis"] += k["us"]; a["rows"] += 1
                if k["stage"] == "unattributed":
                    a["un"] += k["us"]; a["unrows"] += 1; a["names"].add(k["name"])
    print("era      landings  unattr_rows/rows  unattr_distinct  unattr_us/total_us  unattr_us/visible_us")
    for e in ("sglang", "ours", "public", "keell"):
        if e in agg:
            a = agg[e]
            n = sum(1 for c in out["frames"] if era_of[c] == e)
            print("%-8s %8d  %6d/%-6d  %15d  %17.2f%%  %19.2f%%" % (e, n, a["unrows"], a["rows"], len(a["names"]),
                  100 * a["un"] / a["tot"], 100 * a["un"] / a["vis"]))
    print("landings without a profile:", ", ".join("%s(%s)" % m for m in missing) or "none")
    print("landings: %d, with profile: %d" % (len(out["landings"]), len(out["frames"])))


if __name__ == "__main__":
    main()
