#!/usr/bin/env python3
"""Embed data/*.json into page.template.html and write index.html (opens offline from file://).

  python3 build.py            # uses data/times.json, data/loc.json, data/inst.json (if present)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE / 'data'


def load(name, default=None):
    p = D / name
    return json.loads(p.read_text()) if p.exists() else default


def main():
    times = load('times.json')
    if times is None:
        raise SystemExit('data/times.json missing: run collect_times.py first')
    data = {
        'times': times,
        'loc': load('loc.json', {}),
        'inst': load('inst.json', {}),
        'notes': [
            'Times are the judge\'s own per-kernel profile for each scored master landing (H200, one decode step). '
            'Five landings between 10-29 and 10-31 have no profile on record; the page holds the previous frame there and says so.',
            'Heights in the first view compare each group (attention, experts, everything else) with the first landing, '
            'which still ran SGLang\'s own kernels. Finer stages are colour only: on that first landing the projections ran in '
            'shared cuBLAS kernels that cannot be split by stage. Kernels the profile could not place are grey ("unattributed"), never guessed.',
            'Lines of code are counted from the CUDA and Triton source at each landing\'s commit. Library kernels '
            '(cuBLAS, Marlin, FlashAttention) have no lines of ours and are drawn thin with dashed sides.',
            'Instructions executed come from Nsight Compute at six milestone landings, when measured. Until then the depth is fixed.',
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
