#!/usr/bin/env python3
"""decode_layer time anatomy and headline scenarios from a judge profile.

Reads an okc `*.profile.json.gz` for muse_decode_layer, classifies every kernel row of every
case into experts / dense / attention / routing / norms / other, prints per-case component
time and shares, then recomputes the headline exactly as the judge does (per load: weighted
SGLang time over weighted our time; headline = geomean over loads) under scenarios that scale
one component class. A rank column maps the headline onto the 2026-10-26 decode board.

Usage: python3 -I roi_model.py [profile.json.gz]
Default profile: master a903b9d0 (board-equivalent kernels).
"""
import gzip
import json
import math
import os
import re
import sys

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                       'muse_decode_layer', 'hypotheses', 'archive',
                       'combo-measured-rows-slide-batched-preattn-t8-tail', 'a903b9d004d2452d.profile.json.gz')

CLS_ORDER = ['experts', 'dense', 'attention', 'routing', 'norms', 'other']

# decode board on 2026-10-26: (rank, speedup at that rank)
THRESHOLDS = [(1, 2.36), (10, 2.167), (25, 2.034), (50, 1.948), (75, 1.864), (100, 1.80), (150, 1.714),
              (200, 1.628), (250, 1.542), (300, 1.477), (350, 1.426), (400, 1.365), (450, 1.315), (500, 1.268)]


def cls(n):
    s = n.lower()
    if any(k in s for k in ('marlin', 'act_and_mul', 'post_norm_combine', 'post_expert_norm', 'wg_kernel', 'rega')):
        return 'experts'
    # dense before routing: 'dl_pooler_router' contains 'route'
    if any(k in s for k in ('nvjet', 'splitkreduce', 'dl_qkv', 'dl_o_', 'dl_pooler', 'dl_unpool', 'gemm')):
        return 'dense'
    if any(k in s for k in ('align', 'route', 'topk')):
        return 'routing'
    if any(k in s for k in ('flash', 'splitkv', 'sliding', 'pre_attention', 'attn', 'prepare_varlen')):
        return 'attention'
    if 'norm' in s:
        return 'norms'
    return 'other'


ROW = re.compile(r'^(.*?)\s+([\d.]+) (µs|ms)\s+([\d.]+)%\s+(\d+)\s*$')
TOT = re.compile(r'^total\s+([\d.]+) (µs|ms)')


def rank(s):
    for r, v in THRESHOLDS:
        if s >= v:
            return r
    return 999


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    d = json.load(gzip.open(path))
    bm = {b['index']: b for b in d['benchmarks']}
    rows = []
    for p in d['profiles']:
        i = p['index']
        b = bm[i]
        c = p['case']
        comp = {k: 0.0 for k in CLS_ORDER}
        total = None
        for line in p['report'].splitlines():
            m = ROW.match(line)
            if m and not line.startswith(('total', 'GPU busy')):
                comp[cls(m.group(1))] += float(m.group(2)) * (1000 if m.group(3) == 'ms' else 1)
            mt = TOT.match(line)
            if mt:
                total = float(mt.group(1)) * (1000 if mt.group(2) == 'ms' else 1)
        if total is None:
            total = sum(comp.values())
        comp['other'] += max(0, total - sum(comp.values()))
        rows.append(dict(load=c['load'], layer=c['layer'], w=c['weight'], T=c['batch'] * (c.get('q_len') or 1),
                         shape=b['median_ns'] / 1000, sg=b['baseline_ns'] / 1000, total=total, comp=comp))

    print(f'profile: {os.path.relpath(path)}')
    print(f"{'load':10s} {'layer':7s} {'T':>4s} {'ours':>7s} {'sglang':>7s} " + ' '.join(f'{k:>9s}' for k in CLS_ORDER)
          + '   share experts/dense/attn/small')
    for r in rows:
        c = r['comp']
        t = r['total']
        small = c['routing'] + c['norms'] + c['other']
        print(f"{r['load']:10s} {r['layer']:7s} {r['T']:4d} {r['shape']:7.1f} {r['sg']:7.1f} "
              + ' '.join(f"{c[k]:9.1f}" for k in CLS_ORDER)
              + f"   {c['experts'] / t:.0%}/{c['dense'] / t:.0%}/{c['attention'] / t:.0%}/{small / t:.0%}")

    loads = []
    for r in rows:
        if r['load'] not in loads:
            loads.append(r['load'])

    def headline(fn):
        return math.exp(sum(math.log(sum(r['w'] * r['sg'] for r in rows if r['load'] == L)
                                     / sum(r['w'] * fn(r) for r in rows if r['load'] == L)) for L in loads) / len(loads))

    base = headline(lambda r: r['shape'])

    def scen(fx=1.0, fd_small=1.0, fd_big=1.0, fa_dec=1.0, fa_ver=1.0, fs=1.0):
        def f(r):
            c = r['comp']
            save = c['experts'] * (1 - fx) + (c['routing'] + c['norms'] + c['other']) * (1 - fs)
            save += c['dense'] * (1 - (fd_big if r['T'] >= 128 else fd_small))
            if r['layer'] == 'full':
                save += c['attention'] * (1 - (fa_ver if r['load'].startswith('verify') else fa_dec))
            return r['shape'] - save
        return headline(f)

    S = [('experts x0.75', dict(fx=0.75)), ('experts x0.6', dict(fx=0.6)), ('experts x0.5', dict(fx=0.5)),
         ('experts x0.4', dict(fx=0.4)),
         ('dense T<=32 x0.8, T>=128 x0.6', dict(fd_small=0.8, fd_big=0.6)),
         ('dense T<=32 x0.7, T>=128 x0.55', dict(fd_small=0.7, fd_big=0.55)),
         ('full attn: decode x0.76, verify x0.75', dict(fa_dec=0.76, fa_ver=0.75)),
         ('small ops (routing/norms/other) x0.5', dict(fs=0.5)),
         ('experts x0.6 + dense(0.8/0.6)', dict(fx=0.6, fd_small=0.8, fd_big=0.6)),
         ('experts x0.5 + dense(0.8/0.6)', dict(fx=0.5, fd_small=0.8, fd_big=0.6)),
         ('experts x0.5 + dense + small x0.5', dict(fx=0.5, fd_small=0.8, fd_big=0.6, fs=0.5)),
         ('experts x0.5 + dense + attn + small x0.5', dict(fx=0.5, fd_small=0.8, fd_big=0.6, fa_dec=0.76, fa_ver=0.75, fs=0.5)),
         ('experts x0.6 + dense + attn + small x0.6', dict(fx=0.6, fd_small=0.8, fd_big=0.6, fa_dec=0.76, fa_ver=0.75, fs=0.6)),
         ('experts x0.75 + dense + small x0.75', dict(fx=0.75, fd_small=0.8, fd_big=0.6, fs=0.75))]
    print(f"\nbase headline (profile medians) {base:.4f}\n{'scenario':44s} {'headline':>8s} {'x base':>7s} {'~decode rank':>12s}")
    for name, kw in S:
        h = scen(**kw)
        print(f"{name:44s} {h:8.3f} {h / base:7.3f} {rank(h):>12}")


if __name__ == '__main__':
    main()
