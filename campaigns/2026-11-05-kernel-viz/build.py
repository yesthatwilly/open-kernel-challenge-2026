#!/usr/bin/env python3
"""Embed data/*.json into page.template.html and write index.html (opens offline from file://).

  python3 build.py            # uses data/times.json, data/loc.json, data/inst.json (if present)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIRST_RESULT = 'muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.result.json'
FINAL_RESULT = 'muse_decode_layer/hypotheses/BEST.result.json'
# Closing board as read 2026-11-05 13:44Z (README.md "Status: closing board"), and the reads of master's
# final code (CLAIMS.md, DNS entry: 14 reads 2.209-2.284).
BOARD = {'best': 2.284, 'rank': 147, 'entrants': 1299, 'overall_rank': 147, 'overall_entrants': 2014,
         'leader': 2.614, 'hundredth': 2.308, 'start_speedup': 1.316, 'start_rank': 448, 'start_date': '2026-10-26',
         'reads_lo': 2.209, 'reads_hi': 2.284, 'reads_n': 14}
D = HERE / 'data'


def load(name, default=None):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else default


def groups(fr):
    o = {'attention': 0.0, 'experts': 0.0, 'rest': 0.0}
    for k in fr['kernels']:
        o[k['stage'] if k['stage'] in ('attention', 'experts') else 'rest'] += k['us']
    o['rest'] += fr.get('other_us') or 0
    return o


def results(times):
    first = json.loads((REPO / FIRST_RESULT).read_text())
    final = json.loads((REPO / FINAL_RESULT).read_text())
    L = times['landings']
    assert final['job'] == L[-1]['job'] and first['job'] == L[0]['job'], 'result files do not match the first/last landing'
    fa, fb = times['frames'][L[0]['commit']], times['frames'][L[-1]['commit']]
    cases = []
    for i, (s0, s1) in enumerate(zip(first['shapes'], final['shapes'])):
        assert s0['key'].split(' load=')[0] == s1['key'].split(' load=')[0]
        g0, g1 = groups(fa[i]), groups(fb[i])
        tot = sum(g1.values())
        cases.append({'i': i, 'weight': s1['weight'], 'sglang_us': s1['baseline_ns'] / 1e3,
                      'first_us': s0['median_ns'] / 1e3, 'final_us': s1['median_ns'] / 1e3,
                      'first_speedup': s0['speedup'], 'speedup': s1['speedup'], 'sol_us': s1['speed_of_light_ns'] / 1e3,
                      'group_speedup': {k: g0[k] / g1[k] for k in g1}, 'share': {k: g1[k] / tot for k in g1}})
    return {'first': {'speedup': first['speedup'], 'ci': first['speedup_ci'], 'job': first['job']},
            'final': {'speedup': final['speedup'], 'ci': final['speedup_ci'], 'job': final['job']},
            'board': BOARD, 'cases': cases}


def main():
    times = load('times.json')
    if times is None:
        raise SystemExit('data/times.json missing: run collect_times.py first')
    data = {
        'times': times,
        'loc': load('loc.json', {}),
        'inst': load('inst.json', {}),
        'results': results(times),
        'notes': [
            'Times come from the judge itself: for each scored version it reported how long every kernel took in one '
            'token step of each test case, on an NVIDIA H200. For five versions between 10-29 and 10-31 no such breakdown '
            'was kept, so the picture holds the previous version there and says so.',
            'Each version is tied to the judge run that scored it. For eight versions the run id was not recorded and the '
            'match was made by score; the picture flags these.',
            'Kernels the breakdown could not assign to one step are grey ("unattributed") rather than guessed. Version 1 ran '
            'several steps inside shared library calls, which is why heights in the first view are set per group.',
            'Lines of code are counted from the GPU source (CUDA and Triton) at each version. Library kernels such as '
            'NVIDIA\'s cuBLAS contain none of our code and are drawn as thin slivers.',
            'Instruction counts will come from NVIDIA\'s Nsight Compute profiler at six of the versions. Until they are '
            'measured, every block in the second view has the same depth.',
            'Source, scripts and raw data: campaigns/2026-11-05-kernel-viz/ in this repository.',
        ],
    }
    blob = json.dumps(data, separators=(',', ':')).replace('</', '<\\/')
    tpl = (HERE / 'page.template.html').read_text()
    assert '/*DATA*/null' in tpl
    out = tpl.replace('/*DATA*/null', blob)
    (HERE / 'index.html').write_text(out)
    print(f'index.html: {len(out):,} bytes; landings {len(times["landings"])}, '
          f'frames {len(times["frames"])}, loc commits {len(data["loc"])}, inst milestones {len(data["inst"])}')


if __name__ == '__main__':
    main()
