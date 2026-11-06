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
