"""Per-row comparison with each run's own SGLang reference: python3 -I rowcmp2.py OLD.json[.gz] NEW.json[.gz]
ours old -> new (old/new), the run's SGLang time old -> new, and the speedup ratio new/old."""
import gzip
import json
import sys


def load(p):
    return json.load(gzip.open(p) if p.endswith(".gz") else open(p))["result"]


def bench(r):
    out = {}
    for c in r["records"]:
        if c.get("event") == "benchmark":
            case = c["case"]
            out[(case.get("batch"), case.get("q_len", 1), case.get("layer"))] = c
    return out


o, n = load(sys.argv[1]), load(sys.argv[2])
bo, bn = bench(o), bench(n)
for key, cn in bn.items():
    co = bo.get(key)
    if co is None:
        continue
    a, b = co["median_ns"] / 1e3, cn["median_ns"] / 1e3
    sa, sb = a * co["speedup"], b * cn["speedup"]
    print(f"{str(key):26s} ours {a:7.1f} -> {b:7.1f} ({a / b:.3f})  sglang {sa:7.1f} -> {sb:7.1f} ({sa / sb:.3f})  "
          f"speedup x{cn['speedup'] / co['speedup']:.3f}")
print("headline", o["speedup"], "->", n["speedup"])
