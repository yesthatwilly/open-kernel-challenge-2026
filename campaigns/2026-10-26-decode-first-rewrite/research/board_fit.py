#!/usr/bin/env python3
"""How the overall board ranks teams, and what decode_layer speedup moves our overall rank.

Reads `okc leaderboard --json` snapshots (overall + the four kernel-track boards) and:
  1. scores candidate formulas (weighted sum of log-speedups) by Spearman against the
     overall order and by adjacent-pair order violations;
  2. grid-searches the weights that violate the overall order least;
  3. prints the decode_layer speedup needed for overall top-N, holding our other three fixed.

Usage: python3 -I board_fit.py [snapshot_dir]   (default: ./leaderboard-2026-10-26)
Snapshots may be *.json or *.json.gz. Only teams that solved all four boards are fitted.
"""
import gzip
import json
import math
import os
import sys

PROBS = ['muse_attention_decode', 'muse_bf16_proj', 'muse_moe_w4a16', 'muse_decode_layer']


def load(d, name):
    for suffix, opener in (('.json.gz', gzip.open), ('.json', open)):
        p = os.path.join(d, name + suffix)
        if os.path.exists(p):
            with opener(p, 'rt') as f:
                return json.load(f)
    raise FileNotFoundError(f'{name}(.json|.json.gz) in {d}')


def spearman(a, b):
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda i: b[i]))}
    return 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leaderboard-2026-10-26')
    ov = load(d, 'lb_overall')
    boards = {p: load(d, 'lb_' + p) for p in PROBS}
    byteam = {p: {r['team']: r for r in boards[p]['rows']} for p in PROBS}
    full = [(r['rank'], r['team']) for r in ov['rows'] if all(r['team'] in byteam[p] for p in PROBS)]
    ovr = [f[0] for f in full]
    L = {t: [math.log(byteam[p][t]['speedup']) for p in PROBS] for _, t in full}
    print(f'teams on the overall board: {ov["ranked"]}; teams with all four kernel boards: {len(full)}')
    you = ov.get('you', {})
    print(f'us: {you.get("team")} overall rank {you.get("rank")}')
    ours = []
    for p in PROBS:
        y = boards[p].get('you', {})
        ours.append(y.get('speedup'))
        print(f'  {p:24s} {y.get("speedup")}  rank {y.get("rank")}/{boards[p].get("ranked")}')

    def viol(w):
        v = [-(sum(wi * li for wi, li in zip(w, L[t]))) for _, t in full]
        return sum(1 for i in range(len(v) - 1) if v[i] > v[i + 1] + 1e-12), spearman(ovr, v)

    def by_rank(p):
        # order by one board's rank alone (decode-rank-only formula)
        v = [byteam[p][t]['rank'] for _, t in full]
        return spearman(ovr, v)

    print('\nformula (weights att, bf16, moe, decode on log speedup) -> (order violations, spearman)')
    for name, w in [('time-share (.17,.21,.59,1)', (0.17, 0.21, 0.59, 1)), ('equal', (1, 1, 1, 1)),
                    ('decode only', (0, 0, 0, 1)), ('grid best of earlier run', (0.05, 0.05, 0.1, 1))]:
        print(f'  {name:28s} {viol(w)}')
    mean_rank = [sum(byteam[p][t]['rank'] for p in PROBS) / 4 for _, t in full]
    print(f'  {"mean rank":28s} spearman {spearman(ovr, mean_rank):.4f}')
    for p in PROBS:
        print(f'  {"rank on " + p:28s} spearman {by_rank(p):.4f}')

    best = None
    g = [i * 0.02 for i in range(0, 16)]
    for wa in g:
        for wb in g:
            for wm in g:
                vv = viol((wa, wb, wm, 1))
                key = (vv[0], -vv[1])
                if best is None or key < best[0]:
                    best = (key, (wa, wb, wm))
    print(f'\nmin-violation fit: violations {best[0][0]}, spearman {-best[0][1]:.4f}, weights (att, bf16, moe) {best[1]} with decode = 1')

    w = best[1] + (1,)
    allscores = sorted([sum(wi * li for wi, li in zip(w, L[t])) for _, t in full], reverse=True)
    if None in ours:
        print('our speedups missing from a snapshot; skip thresholds')
        return
    print('\ndecode_layer speedup needed, other three held at ours:')
    for target in [25, 50, 100, 150, 200, 300]:
        thr = allscores[target - 1]
        need = math.exp((thr - sum(wi * math.log(s) for wi, s in zip(w[:3], ours[:3]))) / w[3])
        print(f'  overall ~top{target:<4d} (rank among all-4 teams) needs decode {need:.3f}x')


if __name__ == '__main__':
    main()
