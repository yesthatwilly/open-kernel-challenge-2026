"""moe per-row comparison by load, each run against its own SGLang time: python3 -I rowcmp_moe.py OLD NEW"""
import json, gzip, sys
def load(p): return json.load(gzip.open(p) if p.endswith('.gz') else open(p))['result']
def rows(r): return {c['case']['load']: c for c in r['records'] if c.get('event') == 'benchmark'}
a, b = rows(load(sys.argv[1])), rows(load(sys.argv[2]))
for k in a:
    if k in b:
        x, y = a[k], b[k]
        o1, o2, s1, s2 = x['median_ns']/1e3, y['median_ns']/1e3, x['baseline_ns']/1e3, y['baseline_ns']/1e3
        print(f"{k:10s} ours {o1:7.1f} -> {o2:7.1f} ({o1/o2:.3f})  sglang {s1:7.1f} -> {s2:7.1f} ({s1/s2:.3f})  speedup x{(s2/o2)/(s1/o1):.3f}")
