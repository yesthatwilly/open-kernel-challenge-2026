#!/usr/bin/env python3
"""P3b + P3c analysis: Engine A per-PC anatomy on H100 (gpunode06, ncu 2026.2.1, 2026-10-28 06:22-06:26Z).

Usage:  python3 -I p3bc_analysis.py <outdir> [--runs <hopper-probe runs dir>]
Writes: <outdir>/p3bc.json (every number the report quotes) and <outdir>/p3bc_tables.md.  Stdlib only.

Inputs (read-only):
  P3   runs/20261027-203904-gpunode06/out/nodejudge-p3   (ncu.csv, pinned.json, the .txt log)
  P3b  runs/20261027-232220-gpunode06/out/nodejudge-p3b  (ncu.csv, ncu.instances.csv, ncu.source.csv, pass.jsonl,
       ncu-preflight.json, pinned.json, the .txt log)
  P3c  runs/20261027-232407-gpunode06/out/nodejudge-p3c  (same files)
P3's normalization (steps, records per SM, P = distinct_local on routing-exact launches, I_step, tensor-busy) is
research/p3_audit.py, run unchanged as a subprocess on P3 and on P3b.

The source page is SASS only (no line table), so PCs are classified by SASS pattern and position into the regions
of REGION_MAPS below (offsets from the function start; each map was read off the SASS listing and is checked here:
every executed or sampled PC must fall in a named region, each consume body must hold 32 LDSM and 32 / 64 HMMA).
Per-PC 'Instructions Executed' is the SASS-patching count (its own, slower replay pass): spin iterations of the
mbarrier try_wait loops are inflated there. Hardware totals are smsp__inst_executed.sum; the spin share in hardware
terms is hw_total - (instrumented total - instrumented spin), every other region being deterministic.
"""
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DEFAULT = '/home/user/git/takemyjob/prototypes/hopper-probe/runs'
P3_RUN, P3B_RUN, P3C_RUN = '20261027-203904-gpunode06', '20261027-232220-gpunode06', '20261027-232407-gpunode06'
NSM = 132
K = 4096

# ------------------------------------------------------------------------------------------------ region maps
REGION_MAPS = {
    'V1_W13': [
        (0x00000, 0x000a0, 'prologue.entry'), (0x000b0, 0x000b0, 'prologue.griddep_wait'),
        (0x000c0, 0x00110, 'prologue.entry'), (0x00120, 0x01990, 'prologue.build_map'),
        (0x019a0, 0x019c0, 'prologue.entry'), (0x019d0, 0x01b10, 'prologue.mbar_init'),
        (0x01b20, 0x01bf0, 'prologue.syncthreads'), (0x01c00, 0x020a0, 'prologue.split'),
        (0x020b0, 0x020f0, 'prologue.dispatch'), (0x02100, 0x021d0, 'consumer.setup'),
        (0x021e0, 0x022e0, 'consumer.full_wait'), (0x022f0, 0x02410, 'consumer.full_wait.spin'),
        (0x02420, 0x05150, 'consumer.loop'), (0x05160, 0x05230, 'consumer.group_end'),
        (0x05240, 0x07390, 'consumer.fixup'), (0x073a0, 0x07790, 'consumer.epilogue'),
        (0x077a0, 0x077e0, 'consumer.per_stage'), (0x077f0, 0x077f0, 'exit'),
        (0x07800, 0x07e30, 'producer.issue_pre'), (0x07e40, 0x07f70, 'producer.empty_wait'),
        (0x07f80, 0x08080, 'producer.empty_wait.spin'), (0x08090, 0x084c0, 'producer.issue'),
        (0x084d0, 0x08510, 'exit'), (0x08520, 0x08520, 'exit.drain'), (0x08530, 0x09d70, 'cold')],
    'V1_W2': [
        (0x00000, 0x000b0, 'prologue.entry'), (0x000c0, 0x01370, 'prologue.build_map'),
        (0x01380, 0x013a0, 'prologue.entry'), (0x013b0, 0x014f0, 'prologue.mbar_init'),
        (0x01500, 0x015b0, 'prologue.syncthreads'), (0x015c0, 0x01ab0, 'prologue.split'),
        (0x01ac0, 0x01b00, 'prologue.idle_exit'), (0x01b10, 0x01c40, 'consumer.setup'),
        (0x01c50, 0x01d50, 'consumer.full_wait'), (0x01d60, 0x01e80, 'consumer.full_wait.spin'),
        (0x01e90, 0x04ba0, 'consumer.loop'), (0x04bb0, 0x04c80, 'consumer.group_end'),
        (0x04c90, 0x06e70, 'consumer.fixup'), (0x06e80, 0x077c0, 'consumer.epilogue'),
        (0x077d0, 0x07810, 'consumer.per_stage'), (0x07820, 0x07820, 'exit'),
        (0x07830, 0x07f90, 'producer.issue_pre'), (0x07fa0, 0x07fc0, 'exit'),
        (0x07fd0, 0x08150, 'producer.empty_wait'), (0x08160, 0x08260, 'producer.empty_wait.spin'),
        (0x08270, 0x08720, 'producer.issue'), (0x08730, 0x08730, 'exit'), (0x08740, 0x08740, 'exit.drain'),
        (0x08750, 0x08ff0, 'cold')],
    'V4_W13': [
        (0x00000, 0x00120, 'prologue.entry'), (0x00130, 0x00130, 'prologue.griddep_wait'),
        (0x00140, 0x00150, 'prologue.entry'), (0x00160, 0x00260, 'prologue.mbar_init'),
        (0x00270, 0x03640, 'prologue.build_map'), (0x03650, 0x04cc0, 'prologue.split'),
        (0x04cd0, 0x04dc0, 'consumer.setup'), (0x04dd0, 0x04f60, 'consumer.full_wait'),
        (0x04f70, 0x05070, 'consumer.full_wait.spin'), (0x05080, 0x0aaa0, 'consumer.loop'),
        (0x0aab0, 0x0ab70, 'consumer.per_stage'), (0x0ab80, 0x0ac30, 'consumer.group_end'),
        (0x0ac40, 0x0e0f0, 'consumer.fixup'), (0x0e100, 0x0e9d0, 'consumer.epilogue'),
        (0x0e9e0, 0x0ea20, 'consumer.per_stage'), (0x0ea30, 0x0ea30, 'exit'),
        (0x0ea40, 0x0f220, 'producer.issue_pre'), (0x0f230, 0x0f310, 'producer.empty_wait'),
        (0x0f320, 0x0f420, 'producer.empty_wait.spin'), (0x0f430, 0x0fa10, 'producer.issue'),
        (0x0fa20, 0x0fa20, 'exit'), (0x0fa30, 0x0fa30, 'exit.drain'), (0x0fa40, 0x109f0, 'cold')],
    'V4_W2': [
        (0x00000, 0x000d0, 'prologue.entry'), (0x000e0, 0x001e0, 'prologue.mbar_init'),
        (0x001f0, 0x02540, 'prologue.build_map'), (0x02550, 0x03b80, 'prologue.split'),
        (0x03b90, 0x03be0, 'prologue.idle_exit'), (0x03bf0, 0x03d10, 'consumer.setup'),
        (0x03d20, 0x03eb0, 'consumer.full_wait'), (0x03ec0, 0x03fc0, 'consumer.full_wait.spin'),
        (0x03fd0, 0x099f0, 'consumer.loop'), (0x09a00, 0x09ac0, 'consumer.per_stage'),
        (0x09ad0, 0x09b80, 'consumer.group_end'), (0x09b90, 0x0d0c0, 'consumer.fixup'),
        (0x0d0d0, 0x0d8f0, 'consumer.epilogue'), (0x0d900, 0x0d940, 'consumer.per_stage'),
        (0x0d950, 0x0d950, 'exit'), (0x0d960, 0x0e210, 'producer.issue_pre'),
        (0x0e220, 0x0e300, 'producer.empty_wait'), (0x0e310, 0x0e410, 'producer.empty_wait.spin'),
        (0x0e420, 0x0e9b0, 'producer.issue'), (0x0e9c0, 0x0e9c0, 'exit'), (0x0e9d0, 0x0e9d0, 'exit.drain'),
        (0x0e9e0, 0x0f4f0, 'cold')],
    'V6_K1': [
        (0x00000, 0x001b0, 'prologue.entry'), (0x001c0, 0x001c0, 'prologue.griddep_wait'),
        (0x001d0, 0x00220, 'prologue.entry'), (0x00230, 0x01b40, 'prologue.build_map'),
        (0x01b50, 0x01b70, 'prologue.entry'), (0x01b80, 0x01cd0, 'prologue.mbar_init'),
        (0x01ce0, 0x01de0, 'prologue.syncthreads'), (0x01df0, 0x02240, 'prologue.split'),
        (0x02250, 0x02310, 'epilogue.pfull_wait'), (0x02320, 0x02420, 'epilogue.pfull_wait.spin'),
        (0x02430, 0x02bd0, 'epilogue.reduce_store'), (0x02be0, 0x02c20, 'exit'),
        (0x02c30, 0x02c80, 'consumer.bfull_wait'), (0x02c90, 0x02d90, 'consumer.bfull_wait.spin'),
        (0x02da0, 0x02db0, 'consumer.setup'), (0x02dc0, 0x02e70, 'consumer.full_wait'),
        (0x02e80, 0x02fa0, 'consumer.full_wait.spin'), (0x02fb0, 0x05ce0, 'consumer.loop'),
        (0x05cf0, 0x05d80, 'consumer.pfree_wait'), (0x05d90, 0x05eb0, 'consumer.pfree_wait.spin'),
        (0x05ec0, 0x05f90, 'consumer.publish'), (0x05fa0, 0x06000, 'exit'),
        (0x06010, 0x064b0, 'producer.b_copy'), (0x064c0, 0x07000, 'cold'),
        (0x07010, 0x07350, 'producer.issue_pre'), (0x07360, 0x074e0, 'producer.first_full_wait'),
        (0x074f0, 0x07510, 'exit'), (0x07520, 0x07530, 'producer.issue'),
        (0x07540, 0x07600, 'producer.empty_wait'), (0x07610, 0x07710, 'producer.empty_wait.spin'),
        (0x07720, 0x07a00, 'producer.issue'), (0x07a10, 0x07a10, 'exit'), (0x07a20, 0x07a20, 'exit.drain'),
        (0x07a30, 0x08370, 'cold')],
    'V6_K2': [
        (0x00000, 0x00280, 'prologue.entry'), (0x00290, 0x014d0, 'prologue.build_map'),
        (0x014e0, 0x01500, 'prologue.entry'), (0x01510, 0x01630, 'prologue.mbar_init'),
        (0x01640, 0x01690, 'prologue.syncthreads'), (0x016a0, 0x01ea0, 'prologue.split'),
        (0x01eb0, 0x01ec0, 'prologue.dispatch'),
        (0x01ed0, 0x02480, 'epilogue.setup'), (0x02490, 0x02550, 'epilogue.pfull_wait'),
        (0x02560, 0x02660, 'epilogue.pfull_wait.spin'), (0x02670, 0x02f70, 'epilogue.reduce_store'),
        (0x02f80, 0x02fd0, 'exit'), (0x02fe0, 0x03050, 'consumer.bfull_wait'),
        (0x03060, 0x03160, 'consumer.bfull_wait.spin'), (0x03170, 0x03180, 'consumer.setup'),
        (0x03190, 0x03240, 'consumer.full_wait'), (0x03250, 0x03370, 'consumer.full_wait.spin'),
        (0x03380, 0x06050, 'consumer.loop'), (0x06060, 0x060d0, 'consumer.pfree_wait'),
        (0x060e0, 0x06200, 'consumer.pfree_wait.spin'), (0x06210, 0x062e0, 'consumer.publish'),
        (0x062f0, 0x06350, 'exit'), (0x06360, 0x06630, 'producer.issue_pre'),
        (0x06640, 0x06bb0, 'producer.b_copy'), (0x06bc0, 0x07710, 'cold'),
        (0x07720, 0x07740, 'exit'), (0x07750, 0x07770, 'producer.issue'),
        (0x07780, 0x07850, 'producer.empty_wait'), (0x07860, 0x07960, 'producer.empty_wait.spin'),
        (0x07970, 0x07bc0, 'producer.issue'), (0x07bd0, 0x07bd0, 'exit'), (0x07be0, 0x07be0, 'exit.drain'),
        (0x07bf0, 0x080f0, 'cold')],
    'COMBINE': [
        (0x00000, 0x00130, 'combine.entry'), (0x00140, 0x02650, 'combine.load_square_reduce'),
        (0x02660, 0x02730, 'combine.barrier'), (0x02740, 0x04160, 'combine.slots_rsqrt_accumulate'),
        (0x04170, 0x04240, 'combine.store_exit'), (0x04250, 0x10f00, 'cold')],
}
# opcode overrides inside a range (lo, hi, regex on the SASS text, label); ACQBULK is griddepcontrol.wait
OVERRIDES = {
    'V6_K1': [(0x01ce0, 0x01de0, r'BAR\.SYNC', 'prologue.syncthreads')],
    'V6_K2': [(0x01ed0, 0x02480, r'ACQBULK', 'epilogue.griddep_wait'),
              (0x02fe0, 0x03050, r'ACQBULK', 'consumer.griddep_wait'),
              (0x06640, 0x06bb0, r'ACQBULK', 'producer.griddep_wait')],
    'V4_W13': [(0x00270, 0x03640, r'BAR\.SYNC', 'prologue.build_map.barrier')],
    'V4_W2': [(0x001f0, 0x02540, r'BAR\.SYNC', 'prologue.build_map.barrier'),
              (0x03b90, 0x03be0, r'ACQBULK', 'prologue.griddep_wait')],
    'V1_W2': [(0x01ac0, 0x01b00, r'ACQBULK', 'prologue.griddep_wait')],
    'COMBINE': [(0x00000, 0x04240, r'ACQBULK', 'combine.griddep_wait')],
}
ROLE_WARPS = {'V6_K1': {'consumer': 8, 'producer': 1, 'epilogue': 1}, 'V6_K2': {'consumer': 8, 'producer': 1, 'epilogue': 1}}


def kernel_key(kname):
    if 'ea6_kernel' in kname:
        return 'V6_K1' if re.search(r'ea6_kernel<(\(bool\))?(1|true)\b', kname) else 'V6_K2'
    if 'ea_combine_kernel' in kname:
        return 'COMBINE'
    for pat, key in (('ea4_w13', 'V4_W13'), ('ea4_w2', 'V4_W2'), ('ea_w13', 'V1_W13'), ('ea_w2', 'V1_W2')):
        if pat in kname:
            return key
    raise KeyError(kname)


def label_rows(key, rows, exvec):
    """One label per PC. exvec[i] = tuple of PC i's instrumented counts over all launches of the kernel: the one-tile
    and two-tile bodies run on different passes, so their HMMA PCs differ in exec vector; each body is the span
    [its first LDS.128 .. its last HMMA] inside 'consumer.loop'; the rest of the loop range is per-stage control."""
    lab = []
    for r in rows:
        L = None
        for lo, hi, name in REGION_MAPS[key]:
            if lo <= r['off'] <= hi:
                L = name
                break
        lab.append(L)
    for lo, hi, rx, name in OVERRIDES.get(key, []):
        for i, r in enumerate(rows):
            if lo <= r['off'] <= hi and re.search(rx, r['sass']):
                lab[i] = name
    idx = [i for i, L in enumerate(lab) if L == 'consumer.loop']
    if idx:
        groups = {}
        for i in idx:
            if 'HMMA' in rows[i]['sass']:
                groups.setdefault(exvec[i], []).append(i)
        for ev, g in sorted(groups.items(), key=lambda kv: kv[1][0]):
            beg = min(i for i in idx if exvec[i] == ev and 'LDS.128' in rows[i]['sass'] and i < g[0])
            two = any('LDSM.16.M88.4' in rows[i]['sass'] for i in range(beg, g[-1] + 1))
            for i in range(beg, g[-1] + 1):
                if lab[i] == 'consumer.loop':
                    lab[i] = 'consumer.body2' if two else 'consumer.body1'
        for i in idx:
            if lab[i] == 'consumer.loop':
                lab[i] = 'consumer.per_stage'
    return lab


# ------------------------------------------------------------------------------------------------ loaders
def fnum(s):
    s = (s or '').strip().replace(',', '')
    try:
        return float(s)
    except ValueError:
        return None


def load_raw(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    hdr, units = rows[0], rows[1]
    out = []
    for x in rows[2:]:
        out.append({c: x[i] for i, c in enumerate(hdr)})
    return out, dict(zip(hdr, units))


KEYED = re.compile(r'^\s*([-\d.eE+]+)\s*\((.*)\)\s*$', re.S)


def keyed(s):
    """'10843803 (LOP3: 2532600; IMAD: 1641827; ...)' -> (total, {key: value})"""
    m = KEYED.match(s or '')
    if not m:
        return fnum(s), {}
    d = {}
    for part in m.group(2).split(';'):
        if ':' in part:
            k, v = part.rsplit(':', 1)
            d[k.strip()] = fnum(v)
    return fnum(m.group(1)), d


def load_instances(path, cols):
    with open(path, newline='') as f:
        r = csv.reader(f)
        hdr = next(r)
        next(r)
        ix = {c: hdr.index(c) for c in cols if c in hdr}
        out = []
        for x in r:
            out.append({c: keyed(x[i]) for c, i in ix.items()})
    return out


def parse_source(path):
    blocks, cur, hdr = [], None, None
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == 'Kernel Name':
                cur = {'kname': row[1], 'rows': []}
                blocks.append(cur)
                continue
            if row[0] == 'Address':
                hdr = row
                cur['hdr'] = row
                continue
            d = dict(zip(hdr, row))
            rec = {'addr': int(d['Address'], 16), 'sass': d['Source'].strip()}
            for k in hdr[2:]:
                v = fnum(d.get(k, ''))
                rec[k] = v if v is not None else 0.0
            cur['rows'].append(rec)
    for b in blocks:
        a0 = b['rows'][0]['addr']
        for r in b['rows']:
            r['off'] = r['addr'] - a0
    return blocks


def party_rows(path):
    j = json.load(open(path))
    feats, bs = {}, {}
    for rec in j['result']['records']:
        if rec.get('event') == 'benchmark':
            feats[rec['case']['load']] = rec['rep_features']['rows']
            bs[rec['case']['load']] = rec['case']['bs']
    return feats, bs


def nvtx(launch):
    for k, v in launch.items():
        if 'Push/Pop_Range' in k:
            m = re.search(r'nj\|(\w+)\|([^|]+)\|(\d+)', v)
            if m:
                return m.group(1), m.group(2), int(m.group(3))
    raise ValueError('no nj| range')


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(v) if v else None


def mmm(v):
    v = [x for x in v if x is not None]
    return {'med': med(v), 'min': min(v), 'max': max(v), 'n': len(v)} if v else None


STALL_COLS = ['stall_barrier', 'stall_branch_resolving', 'stall_dispatch', 'stall_drain', 'stall_imc', 'stall_lg',
              'stall_long_sb', 'stall_math', 'stall_membar', 'stall_mio', 'stall_misc', 'stall_no_inst',
              'stall_not_selected', 'stall_selected', 'stall_short_sb', 'stall_sleep', 'stall_tex', 'stall_wait']


# ------------------------------------------------------------------------------------------------ helpers
def log_lines(path):
    return open(path, errors='replace').read().splitlines()


def flags_graph(lines, party, shape):
    f = [l for l in lines if re.search(r'PATH %s: FLAGS T=\d+ \(%s\):' % (party, re.escape(shape)), l)]
    g = [l for l in lines if l.startswith('[nodejudge] GRAPH %s %s:' % (party, shape))]
    return f, g


def clock_lines(lines):
    out = {}
    for l in lines:
        m = re.search(r'\[nodejudge\] clock (.+?): SM median (\d+) MHz \(min (\d+)\).*throttle (\w+)', l)
        if m:
            out[m.group(1)] = {'median_mhz': int(m.group(2)), 'min_mhz': int(m.group(3)), 'throttle': m.group(4)}
    return out


def run_p3_audit(csvp, outd, party):
    os.makedirs(outd, exist_ok=True)
    subprocess.run([sys.executable, '-I', os.path.join(HERE, 'p3_audit.py'), csvp, outd, party], check=True)
    return json.load(open(os.path.join(outd, 'tables.json')))


# body instruction -> source operation (consume<2>: ext_prop / scale_prop_rt / mul_bf16x2 / ldsm / mma / lds / other)
def body_op(s):
    s = re.sub(r'^@!?U?P\w+\s+', '', s)
    op = s.split()[0]
    if op.startswith('HMMA'):
        return 'mma'
    if op.startswith('LDSM'):
        return 'ldsm'
    if op.startswith('HMUL2.BF16'):
        return 'mul_bf16x2'
    if op.startswith('LDS'):
        return 'lds (codes + scales)'
    if op == 'PRMT' and re.search(r'0x4[0-3]4[0-3]\b', s):
        return 'scale_prop_rt'
    if op.startswith('LEA') and '0x77807780' in s:
        return 'scale_prop_rt'
    if op.startswith('LOP3') and re.search(r'0x81c081c0|0x1800180\b|0x400040\b|0x80008000\b', s):
        return 'ext_prop'
    if op == 'IMAD.SHL.U32' and re.search(r', (0x8|0x40|0x2)\b', s):
        return 'ext_prop'
    if op == 'SHF.R.U32.HI' and re.search(r', (0x3|0x7)\b', s):
        return 'ext_prop'
    if op in ('BRA', 'BSSY', 'BSYNC', 'WARPSYNC.ALL', 'NOP', 'YIELD') or op.startswith('ISETP') or \
            op.startswith('SYNCS') or op == 'EXIT':
        return 'loop control / sync'
    return 'address math / other'


# ------------------------------------------------------------------------------------------------ main
def main():
    outd = sys.argv[1]
    runs = sys.argv[sys.argv.index('--runs') + 1] if '--runs' in sys.argv else RUNS_DEFAULT
    os.makedirs(outd, exist_ok=True)
    D = {'inputs': {}, 'p3b': {}, 'p3c': {}}
    d3 = os.path.join(runs, P3_RUN, 'out', 'nodejudge-p3')
    d3b = os.path.join(runs, P3B_RUN, 'out', 'nodejudge-p3b')
    d3c = os.path.join(runs, P3C_RUN, 'out', 'nodejudge-p3c')
    D['inputs'] = {'p3': d3, 'p3b': d3b, 'p3c': d3c}

    # ======================================================================== P3b validity
    t3 = run_p3_audit(os.path.join(d3, 'prof', 'ncu.csv'), os.path.join(outd, 'p3_audit_p3'), os.path.join(d3, 'pinned.json'))
    t3b = run_p3_audit(os.path.join(d3b, 'prof', 'ncu.csv'), os.path.join(outd, 'p3_audit_p3b'),
                       os.path.join(d3b, 'pinned.json'))
    L3 = {(l['kernel'], l['shape'], l['rep']): l for l in t3['launches']}
    L3b = {l['id']: l for l in t3b['launches']}
    raw3b, u3b = load_raw(os.path.join(d3b, 'prof', 'ncu.csv'))
    log3, log3b = log_lines(d3 + '.txt'), log_lines(d3b + '.txt')
    V = {'flags_graph': {}, 'pairing': [], 'preflight': None, 'checks': {}, 'clock': {}}
    for shape in ('decode 1', 'decode 16', 'verify 1'):
        f3, g3 = flags_graph(log3, 'pinned', shape)
        fb, gb = flags_graph(log3b, 'pinned', shape)
        V['flags_graph'][shape] = {'flags_identical': bool(f3) and f3 == fb, 'graph_identical': bool(g3) and g3 == gb,
                                   'n_lines': [len(f3), len(g3), len(fb), len(gb)]}
    f3p, _ = party_rows(os.path.join(d3, 'pinned.json'))
    fbp, _ = party_rows(os.path.join(d3b, 'pinned.json'))
    V['rep_features_identical'] = {s: f3p[s] == fbp[s] for s in ('decode 1', 'decode 16', 'verify 1')}
    for x in raw3b:
        i = int(x['ID'])
        l = L3b[i]
        a = L3[(l['kernel'], l['shape'], l['rep'])]
        raw_inst = fnum(x['smsp__inst_executed.sum'])
        raw_tens = fnum(x['sm__pipe_tensor_op_hmma_cycles_active.sum'])
        el_gpc = fnum(x['gpc__cycles_elapsed.avg'])
        el_l1 = fnum(x['l1tex__cycles_elapsed.avg'])
        el_inst = raw_inst / (fnum(x['sm__inst_executed.sum.pct_of_peak_sustained_elapsed']) / 100 * 4 * NSM)
        V['pairing'].append({
            'id': i, 'kernel': l['kernel'], 'shape': l['shape'], 'rep': l['rep'],
            'I_P3': a['I_step_hw'], 'I_P3b_registered': l['I_step_hw'],
            'dI_registered_pct': 100 * (l['I_step_hw'] / a['I_step_hw'] - 1),
            'T_P3': a['tensor_cyc_per_rec'], 'T_P3b_registered': l['tensor_cyc_per_rec'],
            'dT_registered_pct': 100 * (l['tensor_cyc_per_rec'] / a['tensor_cyc_per_rec'] - 1),
            'I_P3b_raw': raw_inst / l['steps'], 'dI_raw_pct': 100 * (raw_inst / l['steps'] / a['I_step_hw'] - 1),
            'T_P3b_raw': raw_tens / 4 / NSM / l['recs_per_sm'],
            'dT_raw_pct': 100 * (raw_tens / 4 / NSM / l['recs_per_sm'] / a['tensor_cyc_per_rec'] - 1),
            'gpc_el_over_inst_pass_el': el_gpc / el_inst, 'l1tex_el_over_inst_pass_el': el_l1 / el_inst})
    P = V['pairing']
    V['pairing_summary'] = {
        'n': len(P), 'registered_outside_3pct': sum(1 for p in P if abs(p['dI_registered_pct']) > 3 or
                                                     abs(p['dT_registered_pct']) > 3),
        'registered_max_abs_dI_pct': max(abs(p['dI_registered_pct']) for p in P),
        'registered_max_abs_dT_pct': max(abs(p['dT_registered_pct']) for p in P),
        'max_abs_dI_minus_dT_pct': max(abs(p['dI_registered_pct'] - p['dT_registered_pct']) for p in P),
        'raw_max_abs_dI_pct': max(abs(p['dI_raw_pct']) for p in P),
        'raw_max_abs_dT_pct': max(abs(p['dT_raw_pct']) for p in P),
        'gpc_el_ratio_range': [min(p['gpc_el_over_inst_pass_el'] for p in P), max(p['gpc_el_over_inst_pass_el'] for p in P)],
        'l1tex_el_ratio_range': [min(p['l1tex_el_over_inst_pass_el'] for p in P),
                                 max(p['l1tex_el_over_inst_pass_el'] for p in P)]}
    pf = json.load(open(os.path.join(d3b, 'prof', 'ncu-preflight.json')))
    V['preflight'] = {k: pf[k] for k in ('version', 'stop', 'missing_required', 'dropped_sections', 'dropped_metrics',
                                         'unreadable')}
    passes = [json.loads(l) for l in open(os.path.join(d3b, 'prof', 'pass.jsonl'))]
    V['checks'] = {'pass_jsonl_check_failures': {p['shape']: p['check_failures'] for p in passes},
                   'passed_lines': [l.strip() for l in log3b if 'passed=' in l],
                   'eager_checks_pass': sum(1 for l in log3b if 'eager reference check pass' in l),
                   'eager_checks_fail': sum(1 for l in log3b if 'eager reference check' in l and 'pass' not in l)}
    V['clock'] = clock_lines(log3b)
    V['cuda_visible_devices'] = [re.search(r'CUDA_VISIBLE_DEVICES=(\S+)', l).group(1) for l in log3b
                                 if 'CUDA_VISIBLE_DEVICES=' in l]
    V['bytes_line'] = [l for l in log3b if l.startswith('[nodejudge] BYTES')]
    lit_ok = (all(v['flags_identical'] and v['graph_identical'] for v in V['flags_graph'].values())
              and V['pairing_summary']['registered_outside_3pct'] == 0 and not pf['stop'] and not pf['missing_required']
              and all(v == 0 for v in V['checks']['pass_jsonl_check_failures'].values())
              and all(c['min_mhz'] >= 1800 and c['throttle'] == 'none' for c in V['clock'].values()))
    V['verdict_registered'] = 'VALID' if lit_ok else 'VOID'
    V['verdict_raw_count_pairing'] = 'PASS' if (V['pairing_summary']['raw_max_abs_dI_pct'] <= 3 and
                                                V['pairing_summary']['raw_max_abs_dT_pct'] <= 3) else 'FAIL'
    D['p3b']['validity'] = V

    # ======================================================================== P3b per-launch metrics
    inst_cols = ['sass__inst_executed_per_opcode', 'sass__inst_executed_per_opcode_pipeline',
                 'sass__inst_executed_per_opcode_with_modifier_all']
    ins3b = load_instances(os.path.join(d3b, 'prof', 'ncu.instances.csv'), inst_cols)
    src3b = parse_source(os.path.join(d3b, 'prof', 'ncu.source.csv'))
    assert len(src3b) == len(raw3b) == len(ins3b)
    launches = []
    for x, ins, blk in zip(raw3b, ins3b, src3b):
        i = int(x['ID'])
        l = L3b[i]
        assert kernel_key(blk['kname']) == kernel_key(x['Kernel Name'])
        assert abs(sum(r['Instructions Executed'] for r in blk['rows']) - fnum(x['inst_executed'])) < 1
        g = lambda c: fnum(x.get(c))
        tot, ops = ins['sass__inst_executed_per_opcode']
        ptot, pipes = ins['sass__inst_executed_per_opcode_pipeline']
        _, mods = ins['sass__inst_executed_per_opcode_with_modifier_all']
        steps = l['steps']
        two_tile_steps = (l['hmma_per_step_proxy'] - 1.0) * steps if not l['v1'] else 0.0
        hw = g('smsp__inst_executed.sum')
        rec = {
            'id': i, 'key': kernel_key(x['Kernel Name']), 'kernel': l['kernel'], 'shape': l['shape'], 'rep': l['rep'],
            'T': l['T'], 'P': l['P_used'], 'dl': l['dl'], 'mx': l['mx'], 'steps': steps, 'recs_per_sm': l['recs_per_sm'],
            'hmma_per_step': fnum(x['sm__inst_executed_pipe_tensor_op_hmma.sum']) / steps,
            'two_tile_steps': two_tile_steps,
            'inst_hw': hw, 'inst_issued': g('smsp__inst_issued.sum'), 'inst_instrumented': tot,
            'issued_over_executed': g('smsp__inst_issued.sum') / hw, 'instrumented_over_hw': tot / hw,
            'I_step_hw': hw / steps,
            'ops_per_step': {k: v / steps for k, v in ops.items()},
            'mods_per_step': {k: v / steps for k, v in mods.items()},
            'pipeline_per_step': {k: v / steps for k, v in pipes.items()},
            'pipeline_excess_per_step': (ptot - tot) / steps,
            'hmul2_per_step': ops.get('HMUL2', 0.0) / steps,
            'fmalite_pipeline_minus_hmul2_per_step': (pipes.get('FMALITE', 0.0) - ops.get('HMUL2', 0.0)) / steps,
            'hw_fma_double_per_step': (g('sm__inst_executed_pipe_fmaheavy.sum') + g('sm__inst_executed_pipe_fmalite.sum')
                                       - g('sm__inst_executed_pipe_fma.sum')) / steps,
            'clk_ghz': g('gpc__cycles_elapsed.avg.per_second') / 1e9 if u3b.get('gpc__cycles_elapsed.avg.per_second') == 'Ghz'
            else g('gpc__cycles_elapsed.avg.per_second') * (1e-3 if u3b.get('gpc__cycles_elapsed.avg.per_second') == 'Mhz' else 1e-9),
        }
        # pipes against SM-active cycles (ratios formed inside one replay pass by ncu)
        pu = {}
        for p in ('alu', 'fma', 'fmaheavy', 'fmalite', 'fma_type_fp16', 'lsu', 'tensor_op_hmma', 'adu', 'cbu', 'xu',
                  'uniform', 'tma', 'tex'):
            pu['inst.' + p] = g('sm__inst_executed_pipe_%s.avg.pct_of_peak_sustained_active' % p)
        for p in ('alu', 'fma', 'fmaheavy', 'fmalite', 'tensor_op_hmma', 'shared', 'tma'):
            pu['cycles.' + p] = g('sm__pipe_%s_cycles_active.avg.pct_of_peak_sustained_active' % p)
        pu['issue_active'] = g('smsp__issue_active.avg.pct_of_peak_sustained_active')
        pu['inst_executed_per_cycle_active_sm'] = g('sm__inst_executed.avg.per_cycle_active')
        rec['pipe_pct_active'] = pu
        rec['pipe_inst_per_step'] = {p: g('sm__inst_executed_pipe_%s.sum' % p) / steps for p in
                                     ('alu', 'fma', 'fmaheavy', 'fmalite', 'lsu', 'tensor_op_hmma', 'adu', 'cbu', 'xu',
                                      'uniform')}
        rec['peaks'] = {'alu_inst_per_cycle_sm': g('sm__inst_executed_pipe_alu.avg.peak_sustained'),
                        'alu_cycles_peak_sm': g('sm__pipe_alu_cycles_active.avg.peak_sustained'),
                        'fmaheavy_cycles_peak_sm': g('sm__pipe_fmaheavy_cycles_active.avg.peak_sustained')}
        # shared memory
        rec['smem'] = {
            'bank_conflicts_ld_per_step': g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum') / steps,
            'bank_conflicts_st_per_step': g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum') / steps,
            'wavefronts_shared_ld_per_step': g('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum') / steps,
            'wavefronts_shared_st_per_step': g('l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum') / steps,
            'wavefronts_lsu_all_per_step': g('l1tex__data_pipe_lsu_wavefronts.sum') / steps,
            'shared_ld_inst_per_step': g('smsp__sass_inst_executed_op_shared_ld.sum') / steps,
            'ldsm_inst_per_step': g('smsp__sass_inst_executed_op_ldsm.sum') / steps,
            'global_ld_inst_per_step': g('smsp__sass_inst_executed_op_global_ld.sum') / steps}
        # resources
        rec['res'] = {k: g(c) for k, c in (('regs', 'launch__registers_per_thread'),
                                           ('regs_alloc', 'launch__registers_per_thread_allocated'),
                                           ('block', 'launch__block_size'), ('grid', 'launch__grid_size'),
                                           ('smem_dyn_B', 'launch__shared_mem_per_block_dynamic'),
                                           ('occ_lim_regs', 'launch__occupancy_limit_registers'),
                                           ('occ_lim_smem', 'launch__occupancy_limit_shared_mem'),
                                           ('occ_lim_warps', 'launch__occupancy_limit_warps'),
                                           ('theor_occ_pct', 'sm__maximum_warps_per_active_cycle_pct'),
                                           ('achieved_occ_pct', 'sm__warps_active.avg.pct_of_peak_sustained_active'),
                                           ('spill_insts', 'sass__inst_executed_register_spilling'),
                                           ('local_spill_req', 'derived__local_spilling_requests'))}
        rec['_blk'] = blk
        launches.append(rec)

    # region labels per kernel (exec vectors across launches)
    bykey = {}
    for rec in launches:
        bykey.setdefault(rec['key'], []).append(rec)
    labels = {}
    for key, lst in bykey.items():
        rows = lst[0]['_blk']['rows']
        assert all([r['sass'] for r in rec['_blk']['rows']] == [r['sass'] for r in rows] for rec in lst)
        exvec = [tuple(int(rec['_blk']['rows'][k]['Instructions Executed']) for rec in lst) for k in range(len(rows))]
        lab = label_rows(key, rows, exvec)
        bad = [('%05x' % r['off']) for L, r in zip(lab, rows) if L is None]
        assert not bad, (key, bad[:5])
        for tag, nh in (('consumer.body1', 32), ('consumer.body2', 64)):
            rr = [r for L, r in zip(lab, rows) if L == tag]
            if rr:
                assert sum('HMMA' in r['sass'] for r in rr) == nh and sum('LDSM' in r['sass'] for r in rr) == 32, (key, tag)
        labels[key] = lab

    def region_tables(rec):
        lab = labels[rec['key']]
        rows = rec['_blk']['rows']
        inst, smp = {}, {}
        st = {}
        for L, r in zip(lab, rows):
            inst[L] = inst.get(L, 0.0) + r['Instructions Executed']
            dr = r['stall_drain']  # a drain stall is a warp past EXIT waiting for its stores: exit.drain wherever it lands
            smp[L] = smp.get(L, 0.0) + r['# Samples'] - dr
            smp['exit.drain'] = smp.get('exit.drain', 0.0) + dr
            d = st.setdefault(L, {})
            for c in STALL_COLS:
                if c != 'stall_drain':
                    d[c] = d.get(c, 0.0) + r[c]
            dd = st.setdefault('exit.drain', {})
            dd['stall_drain'] = dd.get('stall_drain', 0.0) + dr
        for d in st.values():
            for c in STALL_COLS:
                d.setdefault(c, 0.0)
        return inst, smp, st

    for rec in launches:
        inst, smp, st = region_tables(rec)
        steps = rec['steps']
        spin_instr = sum(v for k, v in inst.items() if k.endswith('.spin'))
        det = rec['inst_instrumented'] - spin_instr
        rec['regions_inst_instrumented'] = inst
        rec['regions_samples'] = smp
        rec['regions_stalls'] = st
        rec['spin_instrumented'] = spin_instr
        rec['spin_hw_implied'] = rec['inst_hw'] - det
        hwreg = {k: v for k, v in inst.items() if not k.endswith('.spin')}
        hwreg['all waits: spin iterations (hw-implied)'] = rec['spin_hw_implied']
        rec['regions_per_step_hw'] = {k: v / steps for k, v in hwreg.items()}
        rec['named_share_instrumented'] = 1 - inst.get('cold', 0.0) / rec['inst_instrumented']
        rec['body_share_hw'] = (inst.get('consumer.body1', 0) + inst.get('consumer.body2', 0)) / rec['inst_hw']

    # ---- (a) aggregate per kernel x shape
    def groups():
        g = {}
        for rec in launches:
            g.setdefault((rec['kernel'], rec['shape']), []).append(rec)
        return sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0]))

    A = []
    for (kern, shape), lst in groups():
        regs = sorted({k for rec in lst for k in rec['regions_per_step_hw']})
        A.append({'kernel': kern, 'shape': shape, 'n': len(lst), 'P': [rec['P'] for rec in lst],
                  'I_step_hw': mmm([rec['I_step_hw'] for rec in lst]),
                  'body_share_hw': mmm([rec['body_share_hw'] for rec in lst]),
                  'named_share_instrumented': mmm([rec['named_share_instrumented'] for rec in lst]),
                  'instrumented_over_hw': mmm([rec['instrumented_over_hw'] for rec in lst]),
                  'issued_over_executed': mmm([rec['issued_over_executed'] for rec in lst]),
                  'regions_per_step_hw': {k: mmm([rec['regions_per_step_hw'].get(k, 0.0) for rec in lst]) for k in regs},
                  'pipeline_excess_per_step': mmm([rec['pipeline_excess_per_step'] for rec in lst]),
                  'hmul2_per_step': mmm([rec['hmul2_per_step'] for rec in lst]),
                  'fmalite_pipeline_minus_hmul2_per_step': mmm([rec['fmalite_pipeline_minus_hmul2_per_step'] for rec in lst]),
                  'hw_fma_double_per_step': mmm([rec['hw_fma_double_per_step'] for rec in lst]),
                  'ops_per_step': {k: med([rec['ops_per_step'].get(k, 0.0) for rec in lst]) for k in
                                   sorted({k for rec in lst for k in rec['ops_per_step']})},
                  'pipeline_per_step': {k: med([rec['pipeline_per_step'].get(k, 0.0) for rec in lst]) for k in
                                        sorted({k for rec in lst for k in rec['pipeline_per_step']})},
                  'pipe_pct_active': {k: mmm([rec['pipe_pct_active'][k] for rec in lst]) for k in lst[0]['pipe_pct_active']},
                  'pipe_inst_per_step': {k: med([rec['pipe_inst_per_step'][k] for rec in lst]) for k in lst[0]['pipe_inst_per_step']},
                  'smem': {k: mmm([rec['smem'][k] for rec in lst]) for k in lst[0]['smem']},
                  'res': {k: mmm([rec['res'][k] for rec in lst]) for k in lst[0]['res']},
                  'peaks': lst[0]['peaks']})
    D['p3b']['aggregates'] = A
    # the double charge: which opcode carries pipeline_total - inst_total? regress per launch on HMUL2 count
    xs = [rec['hmul2_per_step'] * rec['steps'] for rec in launches]
    ys = [rec['pipeline_excess_per_step'] * rec['steps'] for rec in launches]
    mx_, my_ = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((a - mx_) * (b - my_) for a, b in zip(xs, ys)) / sum((a - mx_) ** 2 for a in xs)
    D['p3b']['double_charge_fit'] = {'excess_vs_hmul2_slope': slope, 'intercept': my_ - slope * mx_,
                                     'residual_per_step': mmm([(rec['pipeline_excess_per_step'] - rec['hmul2_per_step'])
                                                               for rec in launches]),
                                     'hw_double_minus_hmul2_per_step': mmm([rec['hw_fma_double_per_step'] - rec['hmul2_per_step']
                                                                           for rec in launches])}

    # ---- body opcode budget from the SASS (same per record for every launch of a kernel)
    budget = {}
    for key, lst in bykey.items():
        rows = lst[0]['_blk']['rows']
        lab = labels[key]
        for tag in ('consumer.body1', 'consumer.body2'):
            rr = [r for L, r in zip(lab, rows) if L == tag]
            if not rr:
                continue
            b = {}
            for r in rr:
                b[body_op(r['sass'])] = b.get(body_op(r['sass']), 0) + 1
            per = {}
            for L, r in zip(lab, rows):
                if L in ('consumer.per_stage', 'consumer.full_wait', 'consumer.publish', 'consumer.group_end') and \
                        not L.endswith('.spin'):
                    per[L] = per.get(L, 0)
            budget['%s %s' % (key, tag)] = {'pcs_per_record': len(rr), 'per_step': {k: v / 32 for k, v in b.items()},
                                           'total_per_step': len(rr) / 32}
    D['p3b']['body_budget'] = budget

    # per-record loop overhead outside the body: per-stage + the full-wait first try (instrumented = hw: deterministic)
    overhead = []
    for rec in launches:
        lab = labels[rec['key']]
        rows = rec['_blk']['rows']
        nrec_warp = sum(r['Instructions Executed'] for L, r in zip(lab, rows) if L in ('consumer.body1', 'consumer.body2')
                        and 'HMMA' in r['sass']) / (32.0 * (1.0))
        reg = rec['regions_inst_instrumented']
        overhead.append({'kernel': rec['kernel'], 'shape': rec['shape'], 'rep': rec['rep'],
                         'per_stage_per_step': reg.get('consumer.per_stage', 0) / rec['steps'],
                         'full_wait_first_try_per_step': reg.get('consumer.full_wait', 0) / rec['steps'],
                         'group_end_per_step': reg.get('consumer.group_end', 0) / rec['steps']})
    D['p3b']['loop_overhead'] = overhead

    # ---- (b) stall samples by region; long_scoreboard split; math_pipe_throttle PCs
    B = []
    for (kern, shape), lst in groups():
        tot = {}
        for rec in lst:
            for L, d in rec['regions_stalls'].items():
                t = tot.setdefault(L, {c: 0.0 for c in STALL_COLS})
                for c in STALL_COLS:
                    t[c] += d[c]
        allsmp = sum(sum(d.values()) for d in tot.values())
        lsb = {L: d['stall_long_sb'] for L, d in tot.items() if d['stall_long_sb']}
        mth = {L: d['stall_math'] for L, d in tot.items() if d['stall_math']}
        lsb_t, mth_t = sum(lsb.values()), sum(mth.values())

        def cls(L):
            if L.startswith('consumer.full_wait'):
                return 'consumer full-barrier wait'
            if L.startswith('producer.empty_wait'):
                return 'producer empty-barrier wait'
            if L in ('consumer.epilogue',):
                return 'global load: epilogue gs[le] (+ v_beg / pos_r LDS)'
            if L == 'consumer.fixup':
                return 'global load / atomic: fix-up (ticket ATOMG, partial LDG)'
            if L.startswith('prologue.build_map'):
                return 'global load: build_map topk_ids'
            return 'other'
        split = {}
        for L, v in lsb.items():
            split[cls(L)] = split.get(cls(L), 0.0) + v
        # top PCs (summed over the launches of the group)
        pcs = {}
        lab = labels[lst[0]['key']]
        for rec in lst:
            for k, (L, r) in enumerate(zip(lab, rec['_blk']['rows'])):
                p = pcs.setdefault(k, {'off': r['off'], 'sass': r['sass'], 'region': L, 'lsb': 0.0, 'math': 0.0, 'all': 0.0})
                p['lsb'] += r['stall_long_sb']
                p['math'] += r['stall_math']
                p['all'] += r['# Samples']
        top_lsb = sorted(pcs.values(), key=lambda p: -p['lsb'])[:8]
        top_math = sorted(pcs.values(), key=lambda p: -p['math'])[:8]
        math_by_op = {}
        for p in pcs.values():
            if p['math']:
                op = re.sub(r'^@!?U?P\w+\s+', '', p['sass']).split()[0]
                math_by_op[op] = math_by_op.get(op, 0.0) + p['math']
        reason_tot = {c: sum(d[c] for d in tot.values()) for c in STALL_COLS}
        B.append({'kernel': kern, 'shape': shape, 'samples': allsmp, 'reasons': reason_tot,
                  'lsb_total': lsb_t, 'lsb_share_of_samples': lsb_t / allsmp,
                  'lsb_split': {k: v / lsb_t for k, v in sorted(split.items(), key=lambda kv: -kv[1])},
                  'lsb_by_region': {k: v / lsb_t for k, v in sorted(lsb.items(), key=lambda kv: -kv[1])},
                  'math_total': mth_t, 'math_share_of_samples': mth_t / allsmp,
                  'math_by_region': {k: v / mth_t for k, v in sorted(mth.items(), key=lambda kv: -kv[1])},
                  'math_by_opcode': {k: v / mth_t for k, v in sorted(math_by_op.items(), key=lambda kv: -kv[1])},
                  'samples_by_region': {k: sum(d.values()) / allsmp for k, d in sorted(tot.items(), key=lambda kv: -sum(kv[1].values()))},
                  'top_lsb_pcs': [{k: (('%05x' % v) if k == 'off' else v) for k, v in p.items()} for p in top_lsb],
                  'top_math_pcs': [{k: (('%05x' % v) if k == 'off' else v) for k, v in p.items()} for p in top_math]})
    D['p3b']['stalls'] = B

    # ---- (d) per-PC shared wavefronts actual / ideal on the body's LDS / LDSM PCs; conflicts per two-tile step
    Dd = []
    for (kern, shape), lst in groups():
        lab = labels[lst[0]['key']]
        acc = {}
        for rec in lst:
            for L, r in zip(lab, rec['_blk']['rows']):
                if L not in ('consumer.body1', 'consumer.body2'):
                    continue
                op = re.sub(r'^@!?U?P\w+\s+', '', r['sass']).split()[0]
                if not op.startswith('LDS'):
                    continue
                k = (L, op)
                a = acc.setdefault(k, {'wf': 0.0, 'ideal': 0.0, 'exec': 0.0, 'excess': 0.0, 'pcs': set()})
                a['wf'] += r['L1 Wavefronts Shared']
                a['ideal'] += r['L1 Wavefronts Shared Ideal']
                a['excess'] += r['L1 Wavefronts Shared Excessive']
                a['exec'] += r['Instructions Executed']
                a['pcs'].add(r['off'])
        two_steps = sum(rec['two_tile_steps'] for rec in lst)
        steps = sum(rec['steps'] for rec in lst)
        rows_out = []
        for (L, op), a in sorted(acc.items()):
            if a['exec'] == 0:
                continue
            rows_out.append({'region': L, 'op': op, 'pcs': len(a['pcs']), 'exec': a['exec'],
                             'wavefronts_per_inst': a['wf'] / a['exec'], 'ideal_per_inst': a['ideal'] / a['exec'],
                             'actual_over_ideal': a['wf'] / a['ideal'] if a['ideal'] else None,
                             'excess_wavefronts': a['excess'],
                             'excess_per_step_of_that_body': a['excess'] / (two_steps if L == 'consumer.body2' else
                                                                            (steps - two_steps)) if
                             (two_steps if L == 'consumer.body2' else steps - two_steps) else None})
        Dd.append({'kernel': kern, 'shape': shape, 'two_tile_steps': two_steps, 'steps': steps,
                   'bank_conflicts_ld_per_step': mmm([rec['smem']['bank_conflicts_ld_per_step'] for rec in lst]),
                   'bank_conflicts_ld_per_two_tile_step_upper': (sum(rec['smem']['bank_conflicts_ld_per_step'] * rec['steps']
                                                                     for rec in lst) / two_steps) if two_steps else None,
                   'body_lds': rows_out})
    D['p3b']['smem_pc'] = Dd

    # ======================================================================== P3c
    raw3c, u3c = load_raw(os.path.join(d3c, 'prof', 'ncu.csv'))
    src3c = parse_source(os.path.join(d3c, 'prof', 'ncu.source.csv'))
    f3c, _ = party_rows(os.path.join(d3c, 'pinned.json'))
    log3c = log_lines(d3c + '.txt')
    pf3c = json.load(open(os.path.join(d3c, 'prof', 'ncu-preflight.json')))
    pass3c = [json.loads(l) for l in open(os.path.join(d3c, 'prof', 'pass.jsonl'))]
    C = {'validity': {'preflight': {k: pf3c[k] for k in ('stop', 'missing_required', 'dropped_metrics')},
                      'check_failures': {p['shape']: p['check_failures'] for p in pass3c},
                      'passed_lines': [l.strip() for l in log3c if 'passed=' in l], 'clock': clock_lines(log3c),
                      'bytes_line': [l for l in log3c if l.startswith('[nodejudge] BYTES')],
                      'flags_pinned': [l for l in log3c if 'PATH pinned: FLAGS' in l],
                      'graph_pinned': [l for l in log3c if l.startswith('[nodejudge] GRAPH pinned')],
                      'cuda_visible_devices': [re.search(r'CUDA_VISIBLE_DEVICES=(\S+)', l).group(1) for l in log3c
                                               if 'CUDA_VISIBLE_DEVICES=' in l]}}
    C['validity']['verdict'] = 'VALID' if (not pf3c['stop'] and not pf3c['missing_required'] and
                                           all(p['check_failures'] == 0 for p in pass3c) and
                                           all(c['min_mhz'] >= 1800 for c in C['validity']['clock'].values())) else 'VOID'
    claunch = []
    for x, blk in zip(raw3c, src3c):
        key = kernel_key(blk['kname'])
        assert key == kernel_key(x['Kernel Name']), (blk['kname'][:60], x['Kernel Name'][:60])
        party, shape, rep = nvtx(x)
        lr, dl, mxr = f3c[shape][rep]
        g = lambda c: fnum(x.get(c))
        clk = g('gpc__cycles_elapsed.avg.per_second')
        clk = clk if u3c.get('gpc__cycles_elapsed.avg.per_second') == 'Ghz' else clk / 1000.0
        S = sum(r['# Samples'] for r in blk['rows'])
        W = g('smsp__warps_active.avg.per_cycle_active') * g('smsp__cycles_active.sum')
        rec = {'id': int(x['ID']), 'key': key, 'rep': rep, 'k': dl, 'lr': lr,
               'el_max': g('gpc__cycles_elapsed.max'), 'el_avg': g('gpc__cycles_elapsed.avg'),
               'sm_active_avg': g('sm__cycles_active.avg'), 'sm_active_max': g('sm__cycles_active.max'),
               'dur_us': g('gpu__time_duration.sum'), 'clk_ghz': clk,
               'inst_hw': g('smsp__inst_executed.sum'), 'inst_instrumented': g('inst_executed'),
               'samples': S, 'warp_cycles': W, 'warp_cycles_per_sample': W / S if S else None,
               'smsp_active_per_sample': g('smsp__cycles_active.sum') / S if S else None,
               'grid': g('launch__grid_size'), 'block': g('launch__block_size'), 'regs': g('launch__registers_per_thread'),
               'smem_dyn_B': g('launch__shared_mem_per_block_dynamic'),
               'issue_active_pct': g('smsp__issue_active.avg.pct_of_peak_sustained_active'),
               'warps_active_per_sched': g('smsp__warps_active.avg.per_cycle_active'),
               'warps_eligible_per_sched': g('smsp__warps_eligible.avg.per_cycle_active'),
               'stalls_per_issue': {s: g('smsp__average_warps_issue_stalled_%s_per_issue_active.ratio' % s) for s in
                                    ('long_scoreboard', 'barrier', 'wait', 'membar', 'short_scoreboard', 'math_pipe_throttle',
                                     'no_instruction', 'imc_miss', 'drain', 'branch_resolving', 'dispatch_stall',
                                     'lg_throttle', 'mio_throttle', 'not_selected', 'selected', 'sleeping', 'misc')},
               'dram_read_MB': g('dram__bytes_read.sum'), 'l2_to_sm_MB': g('l1tex__m_xbar2l1tex_read_bytes.sum'),
               '_blk': blk}
        claunch.append(rec)
    # labels
    ckeys = {}
    for rec in claunch:
        ckeys.setdefault(rec['key'], []).append(rec)
    for key, lst in ckeys.items():
        rows = lst[0]['_blk']['rows']
        exvec = [tuple(int(rec['_blk']['rows'][k]['Instructions Executed']) for rec in lst) for k in range(len(rows))]
        lab = label_rows(key, rows, exvec)
        bad = [('%05x' % r['off']) for L, r in zip(lab, rows) if L is None]
        assert not bad, (key, bad[:5])
        labels[key] = lab
    for rec in claunch:
        lab = labels[rec['key']]
        smp, inst, lsb = {}, {}, {}
        for L, r in zip(lab, rec['_blk']['rows']):
            dr = r['stall_drain']
            smp[L] = smp.get(L, 0.0) + r['# Samples'] - dr
            smp['exit.drain'] = smp.get('exit.drain', 0.0) + dr
            inst[L] = inst.get(L, 0.0) + r['Instructions Executed']
            lsb[L] = lsb.get(L, 0.0) + r['stall_long_sb']
        rec['regions_samples'] = smp
        rec['regions_inst'] = inst
        rec['regions_lsb'] = lsb
        rec['no_sm_active_cycles'] = rec['el_max'] - rec['sm_active_max']
        if rec['key'] in ROLE_WARPS:
            # Sampling model (checked: consumer closure within ~3% of sm__cycles_active.avg): one sample per SM every
            # c_s = sm__cycles_active.sum / S cycles, the SM's 4 SMSPs in turn, and on an SMSP its resident warps in turn.
            # v6: 10 warps, SMSP j holds warps {j, j+4, j+8}: n = 3, 3, 2, 2; consumers 0..7, producer 8 (SMSP0, n 3),
            # epilogue 9 (SMSP1, n 3). A warp on an n-warp SMSP is sampled every 4 c_s n cycles while all are resident.
            cs = rec['sm_active_avg'] * NSM / rec['samples']
            inv = {'consumer': 4 / 3 + 4 / 2, 'producer': 1 / 3, 'epilogue': 1 / 3, 'prologue': 4.0, 'exit': 4.0}
            per = {}
            for L, s in smp.items():
                role = L.split('.')[0]
                if L == 'exit.drain':
                    role = 'exit'
                if role not in inv:
                    continue
                per[L] = s * 4 * cs / inv[role] / rec['grid']
            rec['per_warp_cycles'] = per
            rec['cycles_per_sample_sm'] = cs
    # per kernel x k summaries
    Cs = []
    for key in ('V6_K1', 'V6_K2', 'COMBINE'):
        for k in (1, 4):
            lst = [rec for rec in claunch if rec['key'] == key and rec['k'] == k]
            s = {'kernel': key, 'k': k, 'n': len(lst), 'reps': [rec['rep'] for rec in lst]}
            for f in ('el_max', 'el_avg', 'sm_active_avg', 'sm_active_max', 'dur_us', 'clk_ghz', 'inst_hw', 'samples',
                      'issue_active_pct', 'warps_active_per_sched', 'warps_eligible_per_sched', 'dram_read_MB',
                      'l2_to_sm_MB', 'warp_cycles_per_sample', 'smsp_active_per_sample', 'grid', 'block', 'regs',
                      'smem_dyn_B'):
                s[f] = mmm([rec[f] for rec in lst])
            s['el_max_us'] = mmm([rec['el_max'] / (rec['clk_ghz'] * 1e3) for rec in lst])
            s['stalls_per_issue'] = {st: med([rec['stalls_per_issue'][st] for rec in lst]) for st in lst[0]['stalls_per_issue']}
            regs = sorted({L for rec in lst for L in rec['regions_samples']})
            tot = sum(rec['samples'] for rec in lst)
            s['sample_share'] = {L: sum(rec['regions_samples'].get(L, 0) for rec in lst) / tot for L in regs}
            s['lsb_share_of_region'] = {L: (sum(rec['regions_lsb'].get(L, 0) for rec in lst) /
                                            max(1e-9, sum(rec['regions_samples'].get(L, 0) for rec in lst))) for L in regs}
            if key in ROLE_WARPS:
                s['per_warp_cycles'] = {L: mmm([rec['per_warp_cycles'].get(L, 0.0) for rec in lst]) for L in regs
                                        if L in lst[0]['per_warp_cycles']}
                # role residency check: per-warp cycles summed by role vs SM-active
                roles = {}
                for L, v in s['per_warp_cycles'].items():
                    roles.setdefault(L.split('.')[0], 0.0)
                    roles[L.split('.')[0]] += v['med']
                s['role_cycles_sum'] = roles
            # top PCs by samples outside the consume body
            lab = labels[key]
            pcs = {}
            for rec in lst:
                for kk, (L, r) in enumerate(zip(lab, rec['_blk']['rows'])):
                    if L in ('consumer.body1',):
                        continue
                    p = pcs.setdefault(kk, {'off': '%05x' % r['off'], 'sass': r['sass'], 'region': L, 'samples': 0.0,
                                            'reasons': {}})
                    p['samples'] += r['# Samples']
                    for c in STALL_COLS:
                        if r[c]:
                            p['reasons'][c[6:]] = p['reasons'].get(c[6:], 0.0) + r[c]
            s['top_nonloop_pcs'] = sorted(pcs.values(), key=lambda p: -p['samples'])[:14]
            for p in s['top_nonloop_pcs']:
                p['samples_per_launch'] = p['samples'] / len(lst)
            s['instr_per_launch_by_region'] = {L: med([rec['regions_inst'].get(L, 0.0) for rec in lst]) for L in regs}
            Cs.append(s)
    for s in Cs:
        lst = [rec for rec in claunch if rec['key'] == s['kernel'] and rec['k'] == s['k']]
        s['no_sm_active_cycles'] = mmm([rec['no_sm_active_cycles'] for rec in lst])
        s['no_sm_active_us'] = mmm([rec['no_sm_active_cycles'] / (rec['clk_ghz'] * 1e3) for rec in lst])
    fits = {}
    for key in ('V6_K1', 'V6_K2', 'COMBINE'):
        s1 = [s for s in Cs if s['kernel'] == key and s['k'] == 1][0]
        s4 = [s for s in Cs if s['kernel'] == key and s['k'] == 4][0]
        f = {}
        for q in ('el_max', 'el_max_us', 'sm_active_avg', 'sm_active_max', 'dur_us', 'no_sm_active_cycles'):
            c1, c4 = s1[q]['med'], s4[q]['med']
            f[q] = {'k1': c1, 'k4': c4, 'slope_per_expert': (c4 - c1) / 3, 'intercept_k0': (4 * c1 - c4) / 3}
        if 'per_warp_cycles' in s1:
            f['regions_per_warp'] = {}
            for L in sorted(set(s1['per_warp_cycles']) | set(s4['per_warp_cycles'])):
                c1 = (s1['per_warp_cycles'].get(L) or {}).get('med') or 0.0
                c4 = (s4['per_warp_cycles'].get(L) or {}).get('med') or 0.0
                f['regions_per_warp'][L] = {'k1': c1, 'k4': c4, 'slope_per_expert': (c4 - c1) / 3,
                                            'intercept_k0': (4 * c1 - c4) / 3}
        fits[key] = f
    C['fits'] = fits
    C['summaries'] = Cs
    # ---- the P3b metrics on v6 (P3c is VALID): double charge, pipes vs SM-active, shared wavefronts, math stalls
    ins3c = load_instances(os.path.join(d3c, 'prof', 'ncu.instances.csv'),
                           ['sass__inst_executed_per_opcode', 'sass__inst_executed_per_opcode_pipeline'])
    corr = []
    for rec, x, ins in zip(claunch, raw3c, ins3c):
        if rec['key'] not in ('V6_K1', 'V6_K2'):
            continue
        g = lambda c: fnum(x.get(c))
        steps = rec['k'] * (8192 if rec['key'] == 'V6_K1' else 4096) * K / 256
        tot, ops = ins['sass__inst_executed_per_opcode']
        ptot, pipes = ins['sass__inst_executed_per_opcode_pipeline']
        lab = labels[rec['key']]
        wf = ide = 0.0
        mth = {}
        for L, r in zip(lab, rec['_blk']['rows']):
            if L == 'consumer.body1' and re.sub(r'^@!?U?P\w+\s+', '', r['sass']).startswith('LDS'):
                wf += r['L1 Wavefronts Shared']
                ide += r['L1 Wavefronts Shared Ideal']
            if r['stall_math']:
                op = re.sub(r'^@!?U?P\w+\s+', '', r['sass']).split()[0]
                mth[op] = mth.get(op, 0.0) + r['stall_math']
        mt = sum(mth.values()) or 1.0
        alu = sum(v for k2, v in mth.items() if re.match(r'^(LOP3|SHF|LEA|PRMT|ISETP|IADD3|SEL|PLOP3|MOV|VIADD)', k2))
        corr.append({'key': rec['key'], 'k': rec['k'], 'rep': rec['rep'], 'steps': steps,
                     'I_step_hw': rec['inst_hw'] / steps, 'hmul2_per_step': ops.get('HMUL2', 0.0) / steps,
                     'pipeline_excess_per_step': (ptot - tot) / steps,
                     'hw_fma_double_per_step': (g('sm__inst_executed_pipe_fmaheavy.sum') + g('sm__inst_executed_pipe_fmalite.sum')
                                                - g('sm__inst_executed_pipe_fma.sum')) / steps,
                     'alu_inst_per_step': g('sm__inst_executed_pipe_alu.sum') / steps,
                     'pct_active': {p_: g('sm__inst_executed_pipe_%s.avg.pct_of_peak_sustained_active' % p_)
                                    for p_ in ('alu', 'fmaheavy', 'fmalite', 'lsu', 'tensor_op_hmma')},
                     'issue_active': g('smsp__issue_active.avg.pct_of_peak_sustained_active'),
                     'bank_conflicts_ld_per_step': g('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum') / steps,
                     'body_lds_actual_over_ideal': wf / ide if ide else None,
                     'math_on_alu_opcodes': alu / mt,
                     'spin_hw_implied_per_step': (rec['inst_hw'] - (rec['inst_instrumented'] - sum(
                         v for k2, v in rec['regions_inst'].items() if k2.endswith('.spin')))) / steps})
    C['p3b_metrics_on_v6'] = corr
    C['launches'] = [{k: v for k, v in rec.items() if not k.startswith('_')} for rec in claunch]
    # body budget of v6 (same consume<2>)
    for key in ('V6_K1', 'V6_K2'):
        rows = ckeys[key][0]['_blk']['rows']
        lab = labels[key]
        rr = [r for L, r in zip(lab, rows) if L == 'consumer.body1']
        b = {}
        for r in rr:
            b[body_op(r['sass'])] = b.get(body_op(r['sass']), 0) + 1
        per_rec_other = {L: sum(1 for LL in lab if LL == L) for L in ('consumer.per_stage', 'consumer.full_wait',
                                                                      'consumer.publish', 'consumer.pfree_wait')}
        D['p3b']['body_budget']['%s consumer.body1' % key] = {'pcs_per_record': len(rr),
                                                             'per_step': {k: v / 32 for k, v in b.items()},
                                                             'total_per_step': len(rr) / 32,
                                                             'other_loop_pcs_per_record': per_rec_other}
    D['p3c'] = C

    for rec in launches:
        rec.pop('_blk', None)
    D['p3b']['launches'] = launches
    json.dump(D, open(os.path.join(outd, 'p3bc.json'), 'w'), indent=1, default=str)
    write_md(os.path.join(outd, 'p3bc_tables.md'), D)


def f_(v, nd=2):
    if v is None:
        return '-'
    if isinstance(v, dict):
        v = v.get('med')
        if v is None:
            return '-'
    if abs(v) >= 1e5:
        return '%.4g' % v
    if abs(v) >= 100:
        return '%.0f' % v
    return ('%.' + str(nd) + 'f') % v


def rng(d, nd=2):
    if not d:
        return '-'
    return '%s [%s, %s]' % (f_(d['med'], nd), f_(d['min'], nd), f_(d['max'], nd))


def write_md(path, D):
    L = []
    w = L.append
    V = D['p3b']['validity']
    w('# P3b / P3c tables (p3bc_analysis.py)\n')
    w('## P3b validity\n')
    w('- FLAGS/GRAPH vs P3: ' + '; '.join('%s flags %s graph %s' % (s, v['flags_identical'], v['graph_identical'])
                                       for s, v in V['flags_graph'].items()))
    w('- rep_features identical to P3: %s' % V['rep_features_identical'])
    ps = V['pairing_summary']
    w('- registered pairing (pct x 4 x 132 x gpc__cycles_elapsed): %d of %d launches outside 3%%; max |dI| %.2f%%, '
      'max |dT| %.2f%%, max |dI - dT| %.3f%%' % (ps['registered_outside_3pct'], ps['n'], ps['registered_max_abs_dI_pct'],
                                                ps['registered_max_abs_dT_pct'], ps['max_abs_dI_minus_dT_pct']))
    w('- raw-count pairing (smsp__inst_executed.sum, sm__pipe_tensor_op_hmma_cycles_active.sum / 4): max |dI| %.3f%%, '
      'max |dT| %.4f%%' % (ps['raw_max_abs_dI_pct'], ps['raw_max_abs_dT_pct']))
    w('- gpc__cycles_elapsed.avg / inst-pass elapsed: %.4f..%.4f; l1tex__cycles_elapsed.avg / inst-pass elapsed: %.4f..%.4f'
      % tuple(ps['gpc_el_ratio_range'] + ps['l1tex_el_ratio_range']))
    w('- preflight %s; checks %s; clock %s' % (V['preflight'], V['checks']['pass_jsonl_check_failures'], V['clock']))
    w('- **verdict (registered rule): %s**; raw-count pairing: %s\n' % (V['verdict_registered'], V['verdict_raw_count_pairing']))
    w('| kernel | shape | rep | I P3 | I P3b reg | dI reg % | dT reg % | I P3b raw | dI raw % | T raw | gpc el ratio |')
    w('|' + '---|' * 11)
    for p in V['pairing']:
        w('| %s | %s | %d | %.3f | %.3f | %+.2f | %+.2f | %.3f | %+.3f | %.2f | %.4f |' % (
            p['kernel'], p['shape'], p['rep'], p['I_P3'], p['I_P3b_registered'], p['dI_registered_pct'],
            p['dT_registered_pct'], p['I_P3b_raw'], p['dI_raw_pct'], p['T_P3b_raw'], p['gpc_el_over_inst_pass_el']))
    w('\n## (a) instructions per step by region (hardware-equivalent; spin = hw-implied)\n')
    for a in D['p3b']['aggregates']:
        w('### %s %s (P %s)\n' % (a['kernel'], a['shape'], a['P']))
        w('I_step hw %s; body share of hw %s; named share (instrumented) %s; instrumented/hw %s; issued/executed %s' % (
            rng(a['I_step_hw'], 3), rng(a['body_share_hw'], 3), rng(a['named_share_instrumented'], 4),
            rng(a['instrumented_over_hw'], 3), rng(a['issued_over_executed'], 4)))
        w('')
        w('| region | per step med [min, max] |')
        w('|---|---|')
        for k, v in sorted(a['regions_per_step_hw'].items(), key=lambda kv: -(kv[1]['med'] or 0)):
            if v['max'] and v['max'] > 0.0005:
                w('| %s | %s |' % (k, rng(v, 3)))
        w('\nopcode mix per step (instrumented, median): ' + ', '.join('%s %.2f' % (k, v) for k, v in
                                                                      sorted(a['ops_per_step'].items(), key=lambda kv: -kv[1]) if v > 0.01))
        w('\npipeline per step (median): ' + ', '.join('%s %.2f' % (k, v) for k, v in sorted(a['pipeline_per_step'].items(),
                                                                                            key=lambda kv: -kv[1]) if v > 0.01))
        w('\npipeline excess per step %s; HMUL2 per step %s; FMALITE pipeline - HMUL2 %s; hw fmaheavy+fmalite-fma %s\n' % (
            rng(a['pipeline_excess_per_step'], 3), rng(a['hmul2_per_step'], 3),
            rng(a['fmalite_pipeline_minus_hmul2_per_step'], 3), rng(a['hw_fma_double_per_step'], 3)))
        w('pipes, %% of peak over SM-active cycles: ' + ', '.join('%s %s' % (k, rng(v, 1)) for k, v in a['pipe_pct_active'].items()))
        w('\nhw pipe instructions per step: ' + ', '.join('%s %.2f' % (k, v) for k, v in a['pipe_inst_per_step'].items()))
        w('\nshared: ' + ', '.join('%s %s' % (k, rng(v, 3)) for k, v in a['smem'].items()))
        w('\nresources: ' + ', '.join('%s %s' % (k, f_(v['med'], 1)) for k, v in a['res'].items()) + '\n')
    w('\n## double charge fit\n')
    w(json.dumps(D['p3b']['double_charge_fit'], indent=1, default=str))
    w('\n## body budget (per 256-weight step)\n')
    for k, b in D['p3b']['body_budget'].items():
        w('- %s: %d PCs/record = %.3f/step: ' % (k, b['pcs_per_record'], b['total_per_step']) +
          ', '.join('%s %.3f' % (o, v) for o, v in sorted(b['per_step'].items(), key=lambda kv: -kv[1])))
    w('\n## (b) stalls\n')
    for b in D['p3b']['stalls']:
        w('### %s %s: %d samples; long_sb %.1f%%, math %.1f%%\n' % (b['kernel'], b['shape'], b['samples'],
                                                                  100 * b['lsb_share_of_samples'], 100 * b['math_share_of_samples']))
        w('- long_sb split: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in b['lsb_split'].items()))
        w('- long_sb by region: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in b['lsb_by_region'].items() if v > 0.003))
        w('- math by region: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in b['math_by_region'].items() if v > 0.003))
        w('- math by opcode: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in b['math_by_opcode'].items() if v > 0.01))
        w('- samples by region: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in b['samples_by_region'].items() if v > 0.005))
        w('- top long_sb PCs: ' + '; '.join('%s %s [%s] %d' % (p['off'], p['sass'][:40], p['region'], p['lsb'])
                                           for p in b['top_lsb_pcs'][:5]))
        w('- top math PCs: ' + '; '.join('%s %s [%s] %d' % (p['off'], p['sass'][:40], p['region'], p['math'])
                                        for p in b['top_math_pcs'][:5]) + '\n')
    w('\n## (d) shared wavefronts on the body LDS / LDSM PCs\n')
    for d in D['p3b']['smem_pc']:
        w('### %s %s: two-tile steps %.0f of %.0f; bank conflicts (ld) per step %s; per two-tile step (upper bound: all '
          'conflicts charged to two-tile steps) %s\n' % (d['kernel'], d['shape'], d['two_tile_steps'], d['steps'],
                                                        rng(d['bank_conflicts_ld_per_step'], 4),
                                                        f_(d['bank_conflicts_ld_per_two_tile_step_upper'], 4)))
        for r in d['body_lds']:
            w('- %s %s: %d PCs, wavefronts/inst %.3f, ideal/inst %.3f, actual/ideal %s, excess per step %s' % (
                r['region'], r['op'], r['pcs'], r['wavefronts_per_inst'], r['ideal_per_inst'], f_(r['actual_over_ideal'], 4),
                f_(r['excess_per_step_of_that_body'], 4)))
    w('\n## P3c\n')
    C = D['p3c']
    w('validity: %s\n' % C['validity']['verdict'])
    for s in C['summaries']:
        w('### %s k=%d (reps %s)\n' % (s['kernel'], s['k'], s['reps']))
        w('elapsed max %s cyc = %s us at %s GHz; sm-active avg %s, max %s; duration %s us; inst %s; samples %s; issue active %s%%; '
          'warps active / eligible per sched %s / %s; grid %s block %s regs %s smem %s; DRAM read %s MB' % (
              rng(s['el_max'], 0), rng(s['el_max_us'], 2), rng(s['clk_ghz'], 3), rng(s['sm_active_avg'], 0),
              rng(s['sm_active_max'], 0), rng(s['dur_us'], 2), rng(s['inst_hw'], 0), rng(s['samples'], 0),
              rng(s['issue_active_pct'], 1), rng(s['warps_active_per_sched'], 2), rng(s['warps_eligible_per_sched'], 2),
              f_(s['grid'], 0), f_(s['block'], 0), f_(s['regs'], 0), f_(s['smem_dyn_B'], 0), rng(s['dram_read_MB'], 2)))
        w('\nstalls per issue: ' + ', '.join('%s %.2f' % (k, v) for k, v in s['stalls_per_issue'].items() if v and v > 0.01))
        if 'per_warp_cycles' in s:
            w('\n| region | sample share | long_sb share | per-warp cycles med [min, max] |')
            w('|---|---|---|---|')
            for k, v in sorted(s['per_warp_cycles'].items(), key=lambda kv: -(kv[1]['med'] or 0)):
                if v['max'] > 1:
                    w('| %s | %.1f%% | %.0f%% | %s |' % (k, 100 * s['sample_share'][k], 100 * s['lsb_share_of_region'][k], rng(v, 0)))
            w('\nrole sums (per-warp cycles): ' + ', '.join('%s %.0f' % kv for kv in s['role_cycles_sum'].items()))
        else:
            w('\nsample share: ' + ', '.join('%s %.1f%%' % (k, 100 * v) for k, v in s['sample_share'].items()))
        w('\ntop non-body PCs: ' + '; '.join('%s %s [%s] %.0f/launch %s' % (
            p['off'], p['sass'][:38], p['region'], p['samples_per_launch'],
            ','.join('%s:%.0f' % kv for kv in sorted(p['reasons'].items(), key=lambda kv: -kv[1])[:2]))
            for p in s['top_nonloop_pcs'][:10]) + '\n')
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
