"""m0an: SWEEP-pfs1 §3.3 decision from s1pf raw-A/B.jsonl (copy-pooled). usage: python3 -I m0an.py <dir with raw-*.jsonl>"""
import collections
import glob
import json
import math
import random
import sys

D = sys.argv[1]
recs, plans = [], {}
for f in sorted(glob.glob(D + "/raw-*.jsonl")):
    for ln in open(f):
        r = json.loads(ln)
        if "plan" in r:
            plans[(r["plan"], r["arm"])] = ({int(k): v for k, v in r["hist"].items()}, r["nsm"])
        else:
            recs.append(r)
# per (scen, tag, arm, proc, round): mean over copies
cell = collections.defaultdict(list)
for r in recs:
    cell[(r["scen"], r["tag"], r["arm"], r["proc"], r["round"])].append(r["us"])
per = collections.defaultdict(dict)  # (scen, tag, arm) -> {(proc, round): mean}
for (s, t, a, p, rd), v in cell.items():
    per[(s, t, a)][(p, rd)] = sum(v) / len(v)


def med(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def boot(x, f=med, B=3000, seed=1102):
    rnd = random.Random(seed)
    v = sorted(f([x[rnd.randrange(len(x))] for _ in x]) for _ in range(B))
    return v[int(0.025 * B)], v[int(0.975 * B)]


def diff(s, tx, ax, tr, ar):
    a, b = per[(s, tx, ax)], per[(s, tr, ar)]
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    q = [a[k] / b[k] for k in ks]
    return med(d), boot(d), med(q), boot(q), len(ks)


def copyspread(s, t, a):
    by = collections.defaultdict(list)
    for r in recs:
        if (r["scen"], r["tag"], r["arm"]) == (s, t, a):
            by[(r["proc"], r["copy"])].append(r["us"])
    return {k: round(med(v), 2) for k, v in sorted(by.items())}


ITEMS = 96 * 64  # k-blocks per tile (64 GEMM1 + 32 GEMM2 items x 64 k-blocks)
HOST = {"o2": (128, 32), "o3": (128, 64), "o4": (256, 32), "o4b": (256, 64), "o2n2": (128, 32)}
REF = {"o2": ("o1b", "big"), "o3": ("o1b", "big"), "o4": ("M", "big"), "o4b": ("M", "big"), "o2n2": ("M", "big"),
       "o1b": ("M", "big"), "ctl": ("M", "big")}
H200R = {32: 0.18, 64: 0.258}
DLT, CSOLO, NACC = {}, {}, {}
for s in ("t1s", "t1f", "t4s"):
    if (s, "M", "big") not in per:
        continue
    hb, nsm = plans[(s, "big")]
    hs, _ = plans[(s, "small")]
    print(f"== {s}: big plan {dict(sorted(hb.items(), reverse=True))} small plan {dict(sorted(hs.items(), reverse=True))} nsm {nsm}")
    for t, a in (("M", "big"), ("M", "all"), ("M", "small")):
        v = list(per[(s, t, a)].values())
        print(f"   M {a}: med {med(v):.2f} us  copies {copyspread(s, t, a)}")
    dd, ci, q, qci, n = diff(s, "M", "all", "M", "big")
    smallkb = sum(n_ * ITEMS for w, n_ in hs.items()) / nsm
    k = dd / (sum(n_ * H200R.get(w, 0.258) for w, n_ in hs.items()) * ITEMS / nsm)
    CSOLO[s] = {w: k * H200R[w] for w in (32, 64)}
    print(f"   all-big {dd:.2f} us [{ci[0]:.2f}, {ci[1]:.2f}]; small k-blocks/SM {smallkb:.1f}; c_solo/kblock: "
          f"blend {dd / smallkb:.4f} us, N32 {CSOLO[s][32]:.4f}, N64 {CSOLO[s][64]:.4f} (H200 width ratio split)")
    for t in ("ctl", "o1b", "o2", "o3", "o4", "o4b", "o2n2"):
        if (s, t, "big") not in per:
            continue
        rt, ra = REF[t]
        dd, ci, q, qci, n = diff(s, t, "big", rt, ra)
        line = f"   {t:5s} vs {rt}: {dd:+.2f} us [{ci[0]:+.2f}, {ci[1]:+.2f}] ratio {q:.4f} [{qci[0]:.4f}, {qci[1]:.4f}] n {n}"
        if t in HOST:
            hw, ns = HOST[t]
            nh = hb.get(hw, 0)
            cnt = nh * ITEMS / nsm
            if cnt:
                DLT[(s, t)] = (dd / cnt, ci[0] / cnt, ci[1] / cnt)
                line += f" | host{hw} tiles {nh} -> {cnt:.1f} dummy kb/SM -> Delta {dd / cnt:.4f} us/kb [{ci[0] / cnt:.4f}, {ci[1] / cnt:.4f}]"
                line += f" vs c_solo N{ns} {CSOLO[s][ns]:.4f} ({dd / cnt / CSOLO[s][ns]:.2f} x)"
        if t == "o1b":
            nh = hb.get(128, 0)
            NACC[s] = dd / nh if nh else 0.0
            line += f" | NACC1 cost {NACC[s]:+.3f} us per 128-host tile"
        print(line)

# ---- §3.3 projection on judge-like plans (design §1.1 table), d0 0.75 / d1 0.25
PL = {"t1s": [{256: 2, 128: 6, 64: 3, 32: 7}, {256: 5, 128: 1, 32: 11}],
      "t1f": [{256: 2, 128: 5, 64: 2, 32: 6}, {256: 2, 128: 8, 64: 3, 32: 4}],
      "t4s": [{256: 20, 128: 4, 64: 4, 32: 5}, {256: 18, 192: 1, 160: 2, 128: 4, 64: 2, 32: 6}],
      "t4f": [{256: 18, 192: 2, 128: 3, 64: 2, 32: 8}, {256: 18, 160: 3, 128: 3, 64: 3, 32: 4}]}
TROW = {"t1s": 595.0, "t1f": 960.0, "t4s": 2025.0, "t4f": 3480.0}  # H100 PFC2L full-row us (pfsweep 20261101-134606)
NSM = 132
import os
NONACC = os.environ.get("NONACC") == "1"
SRC = {"t1s": "t1s", "t1f": "t1f", "t4s": "t4s", "t4f": "t4s"}  # rate source scenario per row


def delta(s, hw, ns, which=0):
    tag = {(128, 32): "o2", (128, 64): "o3", (256, 32): "o4", (256, 64): "o4b"}[(hw, ns)]
    for sc in (SRC[s], "t1s", "t4s", "t1f"):
        if (sc, tag) in DLT:
            return DLT[(sc, tag)][which]
    return None


def gain(s, plan, hosts753, hosts3012, which=0):
    T753 = s in ("t1s", "t1f")
    hosts = hosts753 if T753 else hosts3012
    cs = CSOLO.get(SRC[s]) or CSOLO["t1s"]
    smalls = [64] * plan.get(64, 0) + [32] * plan.get(32, 0)
    hl = [h for hw in hosts for h in [hw] * plan.get(hw, 0)]
    g = 0.0
    for hw, ns in zip(hl, smalls):
        dl = delta(s, hw, ns, which)
        if dl is None:
            return None
        g += ITEMS / NSM * (cs[ns] - dl)
    if 128 in hosts and not NONACC:
        g -= NACC.get(SRC[s], NACC.get("t1s", 0.0)) * min(plan.get(128, 0), len(smalls))
    return g


VARIANTS = {"S1CF-128": ((128,), (256,)), "S1CF-WIDE": ((128, 256), (256,)), "S1CF-128ONLY": ((128,), (128,))}
print("\n== §3.3 projection (H100; c_solo, Delta, NACC cost measured; small-stream penalty A5-A4 NOT measured, taken 0)")
best = None
for name, (h7, h3) in VARIANTS.items():
    row = {}
    for s in ("t1s", "t1f", "t4s", "t4f"):
        gs = []
        for which in (0, 2, 1):  # central, pessimistic (CI hi Delta), optimistic (CI lo Delta)
            g0 = gain(s, PL[s][0], h7, h3, which)
            g1 = gain(s, PL[s][1], h7, h3, which)
            gs.append(None if g0 is None or g1 is None else 0.75 * g0 + 0.25 * g1)
        row[s] = gs
    if any(row[s][0] is None for s in ("t1s", "t4s")):
        print(f"   {name}: missing rates")
        continue
    pct = {s: [100 * g / TROW[s] if g is not None else float('nan') for g in row[s]] for s in row}
    comb = [math.sqrt((1 + row["t1s"][i] / TROW["t1s"]) * (1 + row["t4s"][i] / TROW["t4s"])) for i in range(3)]
    print(f"   {name}: " + " | ".join(f"{s} {row[s][0]:+.1f} us ({pct[s][0]:+.2f}%) [pess {pct[s][1]:+.2f}, opt {pct[s][2]:+.2f}]"
                                         for s in row) +
          f" | geomean(t1s, t4s) {comb[0]:.4f} [pess {comb[1]:.4f}, opt {comb[2]:.4f}]")
    v = "GO" if comb[0] >= 1.025 and pct["t1f"][0] >= -0.3 else ("SHELF" if comb[0] >= 1.010 else "KILL")
    print(f"   {name}: verdict on central {v}")
    if best is None or comb[0] > best[1]:
        best = (name, comb[0], v)
print(f"\nBEST {best}")
