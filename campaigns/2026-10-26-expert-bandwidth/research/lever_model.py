#!/usr/bin/env python3
"""The decode_layer headline formula and a component-scenario model on a judge profile JSON.

headline = geomean over the 6 loads of (46*sglang_sliding + 16*sglang_full) / (46*ours_sliding + 16*ours_full),
with the weights read from each case's `weight` field. Each case's time is split into component shares from its
kernel table (GPU times; kernels launched with PDL include their wait, so shares of short kernels after a long one are
overstated), and scenarios scale components per case.

Usage: python3 -I lever_model.py <profile.json[.gz]>
"""
import gzip
import json
import math
import re
import sys


def comp_of(name):
    if re.search(r'ea_w13|ea_w2|ea2_|ea3_|marlin|Marlin|wg_kernel', name):
        return 'experts'
    if re.search(r'ea_combine|_post_norm_combine|_post_expert_norm|_hc_align|align2|route_norm_align', name):
        return 'moe_glue'
    if re.search(r'splitkv_attn|skw_attn|skw_pre|_sliding_fused|flash::|_pre_attention|prepare_varlen|FlashAttn', name):
        return 'attention'
    if 'pooler_router' in name:
        return 'pooler_router'
    if re.search(r'_bf16_swapab|nvjet|dl_qkv|dl_o_|dl_unpool|cublas|splitKreduce', name):
        return 'dense'
    return 'small'


def main():
    path = sys.argv[1]
    d = json.load(gzip.open(path, 'rt') if path.endswith('.gz') else open(path))
    cases = []
    for b, p in zip(d['benchmarks'], d['profiles']):
        comps = {}
        for line in p['report'].splitlines():
            m = re.match(r'(.+?)\s+([\d.]+) µs\s+([\d.]+)%\s+(\d+)\s*$', line)
            if m and not line.startswith('total'):
                k = comp_of(m.group(1))
                comps[k] = comps.get(k, 0) + float(m.group(2))
        tot = sum(comps.values())
        c = b['case']
        cases.append(dict(load=c['load'], layer=c['layer'], w=c['weight'], base=b['baseline_ns'] / 1e3,
                          ours=b['median_ns'] / 1e3, frac={k: v / tot for k, v in comps.items()}))

    def headline(fac):
        loads = {}
        for c in cases:
            t = c['ours'] * sum(f * fac(c, k) for k, f in c['frac'].items())
            acc = loads.setdefault(c['load'], [0.0, 0.0])
            acc[0] += c['w'] * c['base']
            acc[1] += c['w'] * t
        return math.exp(sum(math.log(a / b) for a, b in loads.values()) / len(loads)), loads

    base, loads = headline(lambda c, k: 1)
    print(f'reported {d["summary"]["speedup"]:.4f}  model {base:.4f}')
    print('loads: ' + ', '.join(f'{k} {a / b:.3f}' for k, (a, b) in loads.items()))
    for c in cases:
        shares = '  '.join(f'{k} {100 * v:.0f}%' for k, v in sorted(c['frac'].items(), key=lambda kv: -kv[1]))
        print(f'  {c["load"]:9s} {c["layer"]:7s} ours {c["ours"]:7.1f}  {shares}')
    big = lambda c: c['load'] in ('verify 16', 'verify 32')
    full = lambda c: c['layer'] == 'full'
    scen = {
        'experts T=128/256 x0.5': lambda c, k: 0.5 if k == 'experts' and big(c) else 1,
        'experts T<=32 x0.7': lambda c, k: 0.7 if k == 'experts' and not big(c) else 1,
        'full-layer attention x0.6': lambda c, k: 0.6 if k == 'attention' and full(c) else 1,
        'dense T=128/256 x0.6': lambda c, k: 0.6 if k == 'dense' and big(c) else 1,
        'pooler_router + moe_glue + small x0.6': lambda c, k: 0.6 if k in ('pooler_router', 'moe_glue', 'small') else 1,
    }
    print('\nscenarios (headline, multiplier):')
    for name, f in scen.items():
        h, _ = headline(f)
        print(f'  {name:40s} {h:.4f}  x{h / base:.3f}')
    both = lambda c, k: scen['experts T=128/256 x0.5'](c, k) * scen['experts T<=32 x0.7'](c, k)
    h, _ = headline(both)
    print(f'  {"experts at every T (both above)":40s} {h:.4f}  x{h / base:.3f}')


if __name__ == '__main__':
    main()
