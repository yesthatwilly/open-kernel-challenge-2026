"""Per-row comparison of two okc runs: python3 -I rowcmp.py OLD.json[.gz] NEW.json[.gz]"""
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
            key = (case.get("batch"), case.get("q_len", 1), case.get("layer"), case.get("bs"))
            out[key] = c
    return out


o, n = load(sys.argv[1]), load(sys.argv[2])
bo, bn = bench(o), bench(n)
k = next(iter(bn))
pass
for key in bn:
    co, cn = bo.get(key), bn[key]
    def us(c):
        for f in ("median_ns",):
            if c is not None and f in c:
                return c[f]
        return None
    a, b = us(co), us(cn)
    ratio = (a / b) if (a and b) else float("nan")
    print(key, f"old {a / 1e3:8.1f} us  new {b / 1e3:8.1f} us  old/new {ratio:.4f}  sglang/new {cn["speedup"]:.3f}")
print("headline old", o["speedup"], "new", n["speedup"])
