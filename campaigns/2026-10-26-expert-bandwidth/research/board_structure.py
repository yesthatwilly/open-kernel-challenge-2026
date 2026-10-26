#!/usr/bin/env python3
"""How the overall kernel board orders teams, from `okc leaderboard --json` snapshots.

Finding (2026-10-26): every team with a decode_layer run sits in decode-speedup order; teams without one are slotted
in by a composite of their moe / bf16 / attention scores (an Amdahl-style blend, moe weight about 0.48); prefill does
not count. So for a team with a decode run, the overall rank moves only through muse_decode_layer.

Usage: python3 -I board_structure.py [snapshot_dir]   (default: ./leaderboard-2026-10-26T15Z)
"""
import gzip
import json
import math
import os
import sys

P5 = ['muse_attention_decode', 'muse_bf16_proj', 'muse_moe_w4a16', 'muse_decode_layer', 'muse_prefill_layer']


def load(d, name):
    for suffix, opener in (('.json.gz', gzip.open), ('.json', open)):
        p = os.path.join(d, name + suffix)
        if os.path.exists(p):
            with opener(p, 'rt') as f:
                return json.load(f)
    raise FileNotFoundError(name)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           'leaderboard-2026-10-26T15Z')
    ov = load(d, 'lb_overall')
    boards = {p: load(d, 'lb_' + p) for p in P5}
    sp = {p: {r['team']: r['speedup'] for r in boards[p]['rows'] if r.get('rank') is not None} for p in P5}
    rows = sorted([r for r in ov['rows'] if r.get('rank') is not None], key=lambda r: r['rank'])
    teams = [r['team'] for r in rows]
    dec = [sp['muse_decode_layer'].get(t) for t in teams]
    have = [i for i in range(len(teams)) if dec[i] is not None]
    viol = sum(1 for a, b in zip(have, have[1:]) if dec[a] < dec[b])
    print(f'overall rows {len(rows)}; with a decode run {len(have)}; adjacent decode-order violations among them {viol}')
    print('solved counts:', {k: sum(1 for r in rows if r.get("solved") == k) for k in range(1, 6)})

    # Teams without decode: the decode speedups of their nearest decode neighbours bracket the score they were given.
    print('\nteams without a decode run (first 12): overall rank, boards, bracketing decode speedups')
    shown = 0
    moe_only = []
    for i, t in enumerate(teams):
        if dec[i] is not None:
            continue
        up = next((dec[j] for j in range(i - 1, -1, -1) if dec[j]), None)
        dn = next((dec[j] for j in range(i + 1, len(teams)) if dec[j]), None)
        parts = {p[5:9]: sp[p].get(t) for p in P5 if sp[p].get(t)}
        if shown < 12:
            print(f'  {i + 1:5d} {t[:16]:16s} {parts}  between {up} and {dn}')
            shown += 1
        if set(parts) == {'moe_'} and up and dn:
            moe_only.append((parts['moe_'], math.sqrt(up * dn)))
    # Amdahl weight w for moe-only teams: 1/implied = 1 - w (1 - 1/s_moe)
    ws = [(1 - 1 / imp) / (1 - 1 / s) for s, imp in moe_only if s > 1.2]
    if ws:
        ws.sort()
        print(f'\nmoe-only teams: {len(ws)}; implied Amdahl moe weight median {ws[len(ws) // 2]:.3f} '
              f'(p25 {ws[len(ws) // 4]:.3f}, p75 {ws[3 * len(ws) // 4]:.3f})')

    you = ov.get('you', {})
    print(f'\nus: overall rank {you.get("rank")}')
    for p in P5:
        y = boards[p].get('you', {})
        print(f'  {p:24s} {y.get("speedup")}  rank {y.get("rank")}/{boards[p].get("ranked")}')
    dsp = sorted(sp['muse_decode_layer'].values(), reverse=True)
    print('\ndecode speedup -> decode rank today')
    for x in (1.46, 1.51, 1.53, 1.56, 1.62, 1.69, 1.78, 1.86, 1.97, 2.05):
        print(f'  {x:.2f} -> {1 + sum(1 for s in dsp if s > x)}')


if __name__ == '__main__':
    main()
