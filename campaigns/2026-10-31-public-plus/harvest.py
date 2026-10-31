#!/usr/bin/env python3
"""Harvest public-plus okc checks: archive each job and build the per-row control table.

For each job, it archives the raw `okc status --json` record and the profile files to `runs/` (gzipped JSON, plus
the text profile and log), appends a RESULTS.tsv row, and writes a per-row markdown report. The report compares
the A/A reads with each other (spread), with the published public row (RESULTS.md, 10-29) and with our board row
(<problem>/hypotheses/BEST.result.json), and lists the top kernels on both sides.

usage: python3 -I harvest.py <out.md> <arm>=<job> [<arm>=<job> ...]   (run from the repo root; jobs of one problem)
"""
import glob, gzip, json, math, os, re, shutil, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAMP = os.path.join(REPO, "campaigns", "2026-10-31-public-plus")
PUB = os.path.expanduser("~/git/okc-muse-spark-kernels")
SKIP = {"seed"}


def status(job):
    return json.loads(subprocess.run(["okc", "status", job, "--json"], check=True, capture_output=True, text=True).stdout)


def norm(case):
    return {k: str(v) for k, v in (case or {}).items() if k not in SKIP}


def label(case):
    c = norm(case)
    keys = [k for k in ("layer", "load", "m", "bs", "batch", "q_len") if k in c]
    return " ".join(f"{k}={c[k]}" for k in keys) or json.dumps(c)


def top_kernels(rep, n=3):
    out = []
    for line in (rep or "").splitlines()[1:]:
        m = re.match(r"^(.*?)\s{2,}([\d.]+) µs\s+([\d.]+)%\s+(\d+)\s*$", line)
        if m and m.group(1).strip() != "total":
            out.append(f"{m.group(1).strip()[:38]} {float(m.group(2)):.1f}")
        if len(out) >= n:
            break
    return "; ".join(out)


def rows_of(sj):
    recs = (sj.get("result") or {}).get("records") or []
    prof = {r.get("index"): r for r in recs if r.get("event") == "profile"}
    out = []
    for b in recs:
        if b.get("event") != "benchmark":
            continue
        out.append({"case": norm(b.get("case")), "label": label(b.get("case")), "us": (b.get("median_ns") or 0) / 1e3,
                    "base_us": (b.get("baseline_ns") or 0) / 1e3, "speedup": b.get("speedup"),
                    "weight": (b.get("case") or {}).get("weight"),
                    "top": top_kernels((prof.get(b.get("index")) or {}).get("report"))})
    return out


def public_rows(problem):
    txt = open(os.path.join(PUB, "RESULTS.md")).read()
    sec = txt.split(f"## {problem}", 1)[1].split("\n## ", 1)[0]
    out = []
    for line in sec.splitlines():
        m = re.match(r"^\|\s*([a-z_]+=.*?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x\s*\|", line)
        if m:
            case = {}
            for part in m.group(1).split(", "):
                k, _, v = part.partition("=")
                case[k.strip()] = v.strip()
            out.append({"case": case, "sglang_us": float(m.group(2)), "us": float(m.group(3)), "speedup": float(m.group(4))})
    return out


def match(case, cands):
    for c in cands:
        cc = c["case"]
        if all(cc.get(k) == v for k, v in case.items() if k in cc) and all(case.get(k) == v for k, v in cc.items() if k in case):
            return c
    return None


def ours_best(problem):
    p = os.path.join(REPO, problem, "hypotheses", "BEST.result.json")
    if not os.path.exists(p):
        return [], None
    d = json.load(open(p))
    out = []
    for s in d.get("shapes", []):
        top = "; ".join(f"{t[0][:38]} {t[1]:.1f}" for t in (s.get("top_kernels") or [])[:3])
        out.append({"case": norm(s.get("case")), "us": (s.get("median_ns") or 0) / 1e3, "top": top})
    return out, d.get("job")


def archive(sj, job, arm):
    runs = os.path.join(CAMP, "runs")
    os.makedirs(runs, exist_ok=True)
    with gzip.open(os.path.join(runs, f"{job}.okc.json.gz"), "wt") as f:
        json.dump(sj, f)
    dirs = {os.path.dirname(p) if not os.path.isdir(p) else p
            for p in list((sj.get("profile") or {}).values()) + [(sj.get("output") or {}).get("log")] if p}
    for d in dirs:
        for src in glob.glob(os.path.join(d, f"*{job}*")):
            ext = src.rsplit(".", 1)[-1]
            dst = os.path.join(runs, f"{job}.{arm}.{ {'txt': 'profile.txt', 'json': 'profile.json'}.get(ext, ext)}")
            if ext == "json":
                with open(src, "rb") as fi, gzip.open(dst + ".gz", "wb") as fo:
                    fo.write(fi.read())
            else:
                shutil.copyfile(src, dst)


def main():
    out_md, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    reads = []
    for arm, job in pairs:
        sj = status(job)
        if sj.get("status") != "done":
            sys.exit(f"{job} not done: {sj.get('status')}")
        archive(sj, job, arm)
        reads.append((arm, job, sj))
    problem = reads[0][2]["board"]
    pub, (ours, ours_job) = public_rows(problem), ours_best(problem)
    heads = [r[2]["result"]["speedup"] for r in reads]
    gm = math.exp(sum(math.log(h) for h in heads) / len(heads))
    s_h = (max(heads) - min(heads)) / gm if len(heads) > 1 else float("nan")
    per = [rows_of(r[2]) for r in reads]
    tsv = os.path.join(CAMP, "RESULTS.tsv")
    with open(tsv, "a") as f:
        for (arm, job, sj), rows in zip(reads, per):
            r = sj["result"]
            ci = r.get("speedup_ci") or [None, None]
            row_us = ", ".join(f"{x['label']}:{x['us']:.2f}" for x in rows)
            when = __import__("datetime").datetime.fromtimestamp(sj["finished_at"], __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
            f.write(f"{when}\t{job}\t{problem}\t{arm}\t{r['speedup']:.4f}\t[{ci[0]:.4f}, {ci[1]:.4f}]\t{row_us}\t-\t"
                    f"control read; cluster {sj.get('cluster')}; prepare {r['records'][0].get('prepare_s', 0):.0f} s\n")
    L = [f"# {problem}: same-day public control ({', '.join(a for a, _, _ in reads)})\n",
         f"Reads: " + ", ".join(f"{a} `{j}` {s['result']['speedup']:.4f} [{s['result']['speedup_ci'][0]:.4f}, "
                                f"{s['result']['speedup_ci'][1]:.4f}] on {s.get('cluster')}" for a, j, s in reads),
         f"\nControl mean (geomean) **{gm:.4f}**, headline spread s_h = {s_h * 100:.2f}%. Public published (10-29) "
         f"row: see RESULTS.md. Our board BEST job `{ours_job}`.\n",
         "| row | w | public 10-29 us | " + " | ".join(f"{a} us" for a, _, _ in reads) +
         " | mean us | s_r % | today vs 10-29 | ours BEST us | ours/theirs (today) | their top kernels (today) | our top kernels (BEST) |",
         "|---|---:|---:|" + "---:|" * len(reads) + "---:|---:|---:|---:|---:|---|---|"]
    for i, base in enumerate(per[0]):
        us = [p[i]["us"] for p in per]
        mean = sum(us) / len(us)
        s_r = (max(us) - min(us)) / mean * 100 if len(us) > 1 else float("nan")
        pr, ob = match(base["case"], pub), match(base["case"], ours)
        L.append(f"| {base['label']} | {base['weight']} | {pr['us'] if pr else '-'} | " + " | ".join(f"{u:.2f}" for u in us) +
                 f" | {mean:.2f} | {s_r:.1f} | {(mean / pr['us']) if pr else float('nan'):.3f} | {ob['us'] if ob else float('nan'):.2f} | "
                 f"{(ob['us'] / mean) if ob else float('nan'):.3f} | {base['top']} | {ob['top'] if ob else '-'} |")
    open(out_md, "w").write("\n".join(L) + "\n")
    print(problem, f"mean {gm:.4f} s_h {s_h * 100:.2f}%", out_md)


if __name__ == "__main__":
    main()
