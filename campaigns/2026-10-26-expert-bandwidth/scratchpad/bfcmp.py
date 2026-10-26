import json, sys, gzip, os
def load(p):
    op = gzip.open if p.endswith('.gz') else open
    d = json.load(op(p))
    if 'result' in d and isinstance(d['result'], dict) and 'records' in d['result']:
        recs = d['result']['records']
        out = {}
        for r in recs:
            if r.get('event') != 'benchmark': continue
            c = r['case']; out[(c['load'], c['layer'], c['m'])] = (r['median_ns']/1e3, r['baseline_ns']/1e3, r['speedup'])
        return d['result'].get('speedup'), out
    out = {}
    for s in d['shapes']:
        c = s['case']; out[(c['load'], c['layer'], c['m'])] = (s['median_ns']/1e3, s['baseline_ns']/1e3, s['speedup'])
    return d.get('speedup'), out
a_sp, a = load(sys.argv[1]); b_sp, b = load(sys.argv[2])
print('A', a_sp, 'B', b_sp)
for k in a:
    if k in b:
        ra = a[k][2]; rb = b[k][2]
        flag = '  <<' if rb/ra > 1.03 else ('  >>' if rb/ra < 0.97 else '')
        print(f"{k[0]:10s} {k[1]:9s} m={k[2]:4} A={a[k][0]:6.2f}(x{ra:.3f}) B={b[k][0]:6.2f}(x{rb:.3f}) B/A={rb/ra:.3f}{flag}")
