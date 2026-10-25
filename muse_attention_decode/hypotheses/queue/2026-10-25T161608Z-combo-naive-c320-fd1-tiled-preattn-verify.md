---
problem: muse_attention_decode
slug: combo-naive-c320-fd1-tiled-preattn-verify
created: 2026-10-25T161608Z
base: master
area: "combination (naive top-row stack: full decode_1 chunk320 + verify tiled pre-attention)"
change: >
  COMBINATION 1 of 3, the naive top-row stack. Built on current master 7e12891 (blob 5cf8fae;
  restack-sliding-decode-sk-window-on-cluster-master, 1.5017 scored). Re-based against master's
  two runs (252abb7f, 917a9689) only two sources beat master on any row today: chunk320 on full
  decode_1 (-13%) and the group-tiled pre-attention on sliding verify_32 (-4%, twice). Every other
  'strong' source fell back to master in its own run or was measured against a regressed base (see
  notes). This combination routes each of those rows to its winner, each behind its own gate, and
  leaves every other class on master byte for byte. SOURCE A = full-decode1-cuda-splitkv-chunk320
  (branch hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job
  8d4771feca9b4e18: full decode_1 19.49 us = splitkv_attn_kernel<320> 11.0 +
  splitkv_combine_kernel<320,16> 4.5 + _pre_attention_kernel 1.3; master's two-level stream-K is
  22.40 us in 252abb7f and 22.53 in 917a9689, so 0.87x of master on the same row, paired vs SGLang
  1.604 vs 1.393/1.395). It was never compared with the two-level kernel: it harvested neutral on
  an older base and s3 landed after it. Master's _SPLITKV_CUDA is byte-identical to that run's
  base, so the CUDA ports cleanly. Port: the templated splitkv_attn_kernel<CHUNK> and
  splitkv_combine_kernel<CHUNK, NW> with launch_attn<320> / launch_combine<320,16> (512-thread
  combine), the strided issue_meta loop, the __host__ __device__ constexpr smem helpers (audit
  fix), instantiated ONLY at <320> and <320,16>, as a NEW load_inline module muse_full_decode_c320
  (TORCH_CUDA_ARCH_LIST 9.0a, -O3 -lineinfo, -std=c++20 in extra_cuda_cflags, binaryVersion == 90
  check, print regs/local/smem). Master's v1 module and its <1024> code stay untouched. Python:
  _full_decode_c320(data, q, qkv) = master _pre_attention_kernel (unpaired, exactly as master's
  batch-1 branch) then c320 A + combine, max_chunks = ceil(page_table.shape[1] / 320), grid 2 x
  SM. Gate class FD1 = config window == 0, q_len 1, batch 1, tp8, d 64, page 1 (the same predicate
  as master's _shape_gate at B=1). GATE A (FD1): (i) correctness: the chunk320 run's batch-1 cases
  (lens [80000] and [5000], kv 1 and 0.25, grid device and 3) plus 1 / 63 / 64 / 4097 keys, rel L2
  < 2e-2 vs master's FP32 _splitkv_reference AND < 5e-3 vs master's two-level output on the same
  inputs, finite, cache rows equal to master's pre-attention rows (same kernel); (ii) two-state
  alternating capture-once replay (A B B A A B, refilled inputs, NaN-poisoned outputs) at 80000
  and 4097 keys; (iii) speed: interleaved cold-L2 CUDA-graph replay (256 MB flush, empty graph
  subtracted, 41 reps, median) of [pre-attn + c320 A + combine] vs master's [pre-attn + two-level
  <4,true,true> grid SM] at B=1 x 80000; route only if c320 / master <= 0.98 (expected ~0.87).
  Print 'PATH CHOSEN: full decode_1 [A chunk320]: <route|master two-level> (c320 X us vs two-level
  Y us, ratio R)'. SOURCE B = preattention-tiled-gate-large-tokens (branch
  hyp/muse_attention_decode/preattention-tiled-gate-large-tokens, impl 880b01b, job
  c2ebf6ed05824692; parent preattention-group-tiled-programs impl 9c4ab08, job 877731a3):
  _pre_attention_tiled_kernel verbatim (grid (tokens, Hkv), num_warps 4, program (t, g) normalizes
  the G = Hq/Hkv query heads of group g as one [G, 64] tile plus key head g, same tl.cos/tl.sin
  and rounding points as master, PAIRED and unpaired q layouts, FP8 K/V rows / k_scale, v_scale at
  cache_loc). It measured _pre_attention_kernel 4.1-4.2 -> 2.7 us at sliding verify_32 (shape
  28.70 -> 27.55, 1.0385) and 2.8 -> 2.0 us at sliding verify_16; master still shows 4.2 / 2.8 us
  there. Use it at kernel_fn's single _pre_attention_kernel launch (before the splitkv / pseudo /
  FA3 branches) only for the verify classes its gate routes; decode and every self-check helper
  keep master's kernel. Gate classes (config only, q_len 8): SV1, SV16 (B 2..16), SV32 (B 17..32)
  on sliding (unpaired; SV1 feeds the PSEUDO route), FV1 (unpaired), FV16, FV32 (paired). GATE B
  per class: (i) exactness vs master's kernel on the same inputs at that class's representative
  batch (1/16/32 x 8 tokens, plus the test shapes batch 5 / 3 / 4 sliding verify page 1 and kv
  0.015625): FP8 q bytes and K/V cache bytes identical except <= 1% of bytes that differ by one
  E4M3 ulp (master's own Triton fast sin/cos is the same code), PAIRED zero halves exactly zero,
  dequantized rel L2 < 1e-3; (ii) speed: interleaved cold-L2 graph replay of the pre-attention
  launch alone (tiled vs master, 41 reps, empty subtracted) AND of the whole class step (pre-attn
  + that class's downstream path as kernel_fn would run it), route only if the whole-step ratio
  tiled / master <= 0.98. Print one 'PATH CHOSEN: pre-attention <class> [B tiled]: <route|master>
  (pre X vs Y us, step X vs Y us)' per class. Expected: SV32 routes, SV16 marginal, FV16/FV32 stay
  master (paired layout gained only 0.1 us), SV1/FV1 stay master (8 tokens). PHASES AFTER MASTER:
  c320 build ~40 s (deadline 120 s); GATE A ~10 s (deadline 40 s); GATE B ~15 s incl. Triton JIT
  of the tiled kernel (two PAIRED specializations x ROPE on/off) (deadline 40 s); total ~65 s
  added. IMPORT BUDGET (no prepare() on this problem: everything runs at import and counts toward
  the 15-min run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s
  each). Master's import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1,
  _ps_setup, _skf_setup, diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its
  own thresholds. Every added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock
  and is SKIPPED (master kept for its classes, PATH CHOSEN says so) if the clock is past 420 s
  when it would start or past its own deadline when it finishes. RULES (every path): CUDA-graph
  capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config
  only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable
  Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16
  out; every key of the context attended; no child processes, no timing in kernel_fn. The judge
  prepares TWO states and rotates them: no module/static global may hold a device pointer into a
  state's tensors, a TMA map or a state-sized buffer (the c320 path has no tickets; its workspaces
  are torch.empty per call, sized from page_table.shape[1] and batch). Route tables are frozen
  (MappingProxyType / module bools set once at import) before any capture.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: chunk320 19.49 us vs master 22.40/22.53 us on this row (A 11.0 + combine 4.5 + pre 1.3); -2.9 us x 16 = 46 us of the 905 us decode_1 load, ~+0.85% headline
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: tiled pre-attention 4.2 -> 2.7 us measured twice; shape 29.2 -> ~28.1 us; ~+0.14% headline
  - shapes: "*layer=sliding load=verify_16 *"
    direction: flat
    why: kernel 2.8 -> 2.0 us but the shape moved only 22.67 -> 22.74 in c2ebf6ed; routed only if the whole-step gate says <= 0.98
  - shapes: "*layer=sliding load=decode_* *"
    direction: flat
    why: master byte for byte (cluster C=16, Triton, sk window)
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: flat
    why: master v2 stream-K (+skF gate) unchanged
  - shapes: "*layer=full load=verify_* *"
    direction: flat
    why: paired FA3; tiled expected to be gated off (0.1 us at FV16)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-naive-c320-fd1-tiled-preattn-verify

## Change

COMBINATION 1 of 3, the naive top-row stack. Built on current master 7e12891 (blob 5cf8fae; restack-sliding-decode-sk-window-on-cluster-master, 1.5017 scored). Re-based against master's two runs (252abb7f, 917a9689) only two sources beat master on any row today: chunk320 on full decode_1 (-13%) and the group-tiled pre-attention on sliding verify_32 (-4%, twice). Every other 'strong' source fell back to master in its own run or was measured against a regressed base (see notes). This combination routes each of those rows to its winner, each behind its own gate, and leaves every other class on master byte for byte. SOURCE A = full-decode1-cuda-splitkv-chunk320 (branch hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320, impl 183eb55, job 8d4771feca9b4e18: full decode_1 19.49 us = splitkv_attn_kernel<320> 11.0 + splitkv_combine_kernel<320,16> 4.5 + _pre_attention_kernel 1.3; master's two-level stream-K is 22.40 us in 252abb7f and 22.53 in 917a9689, so 0.87x of master on the same row, paired vs SGLang 1.604 vs 1.393/1.395). It was never compared with the two-level kernel: it harvested neutral on an older base and s3 landed after it. Master's _SPLITKV_CUDA is byte-identical to that run's base, so the CUDA ports cleanly. Port: the templated splitkv_attn_kernel<CHUNK> and splitkv_combine_kernel<CHUNK, NW> with launch_attn<320> / launch_combine<320,16> (512-thread combine), the strided issue_meta loop, the __host__ __device__ constexpr smem helpers (audit fix), instantiated ONLY at <320> and <320,16>, as a NEW load_inline module muse_full_decode_c320 (TORCH_CUDA_ARCH_LIST 9.0a, -O3 -lineinfo, -std=c++20 in extra_cuda_cflags, binaryVersion == 90 check, print regs/local/smem). Master's v1 module and its <1024> code stay untouched. Python: _full_decode_c320(data, q, qkv) = master _pre_attention_kernel (unpaired, exactly as master's batch-1 branch) then c320 A + combine, max_chunks = ceil(page_table.shape[1] / 320), grid 2 x SM. Gate class FD1 = config window == 0, q_len 1, batch 1, tp8, d 64, page 1 (the same predicate as master's _shape_gate at B=1). GATE A (FD1): (i) correctness: the chunk320 run's batch-1 cases (lens [80000] and [5000], kv 1 and 0.25, grid device and 3) plus 1 / 63 / 64 / 4097 keys, rel L2 < 2e-2 vs master's FP32 _splitkv_reference AND < 5e-3 vs master's two-level output on the same inputs, finite, cache rows equal to master's pre-attention rows (same kernel); (ii) two-state alternating capture-once replay (A B B A A B, refilled inputs, NaN-poisoned outputs) at 80000 and 4097 keys; (iii) speed: interleaved cold-L2 CUDA-graph replay (256 MB flush, empty graph subtracted, 41 reps, median) of [pre-attn + c320 A + combine] vs master's [pre-attn + two-level <4,true,true> grid SM] at B=1 x 80000; route only if c320 / master <= 0.98 (expected ~0.87). Print 'PATH CHOSEN: full decode_1 [A chunk320]: <route|master two-level> (c320 X us vs two-level Y us, ratio R)'. SOURCE B = preattention-tiled-gate-large-tokens (branch hyp/muse_attention_decode/preattention-tiled-gate-large-tokens, impl 880b01b, job c2ebf6ed05824692; parent preattention-group-tiled-programs impl 9c4ab08, job 877731a3): _pre_attention_tiled_kernel verbatim (grid (tokens, Hkv), num_warps 4, program (t, g) normalizes the G = Hq/Hkv query heads of group g as one [G, 64] tile plus key head g, same tl.cos/tl.sin and rounding points as master, PAIRED and unpaired q layouts, FP8 K/V rows / k_scale, v_scale at cache_loc). It measured _pre_attention_kernel 4.1-4.2 -> 2.7 us at sliding verify_32 (shape 28.70 -> 27.55, 1.0385) and 2.8 -> 2.0 us at sliding verify_16; master still shows 4.2 / 2.8 us there. Use it at kernel_fn's single _pre_attention_kernel launch (before the splitkv / pseudo / FA3 branches) only for the verify classes its gate routes; decode and every self-check helper keep master's kernel. Gate classes (config only, q_len 8): SV1, SV16 (B 2..16), SV32 (B 17..32) on sliding (unpaired; SV1 feeds the PSEUDO route), FV1 (unpaired), FV16, FV32 (paired). GATE B per class: (i) exactness vs master's kernel on the same inputs at that class's representative batch (1/16/32 x 8 tokens, plus the test shapes batch 5 / 3 / 4 sliding verify page 1 and kv 0.015625): FP8 q bytes and K/V cache bytes identical except <= 1% of bytes that differ by one E4M3 ulp (master's own Triton fast sin/cos is the same code), PAIRED zero halves exactly zero, dequantized rel L2 < 1e-3; (ii) speed: interleaved cold-L2 graph replay of the pre-attention launch alone (tiled vs master, 41 reps, empty subtracted) AND of the whole class step (pre-attn + that class's downstream path as kernel_fn would run it), route only if the whole-step ratio tiled / master <= 0.98. Print one 'PATH CHOSEN: pre-attention <class> [B tiled]: <route|master> (pre X vs Y us, step X vs Y us)' per class. Expected: SV32 routes, SV16 marginal, FV16/FV32 stay master (paired layout gained only 0.1 us), SV1/FV1 stay master (8 tokens). PHASES AFTER MASTER: c320 build ~40 s (deadline 120 s); GATE A ~10 s (deadline 40 s); GATE B ~15 s incl. Triton JIT of the tiled kernel (two PAIRED specializations x ROPE on/off) (deadline 40 s); total ~65 s added. IMPORT BUDGET (no prepare() on this problem: everything runs at import and counts toward the 15-min run; master job 252abb7f wall was 470 s with four load_inline builds of ~41-50 s each). Master's import sequence (v1 build + self-check, v2/PSEUDO build, _sk_validate_b1, _ps_setup, _skf_setup, diagnostics, _slide_init, _sd_setup) runs FIRST and unchanged, with its own thresholds. Every added phase runs after _sd_setup, checks a module-level IMPORT_T0 clock and is SKIPPED (master kept for its classes, PATH CHOSEN says so) if the clock is past 420 s when it would start or past its own deadline when it finishes. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key of the context attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (the c320 path has no tickets; its workspaces are torch.empty per call, sized from page_table.shape[1] and batch). Route tables are frozen (MappingProxyType / module bools set once at import) before any capture.

## Expected

- `*layer=full load=decode_1 *` faster: chunk320 19.49 us vs master 22.40/22.53 us on this row (A 11.0 + combine 4.5 + pre 1.3); -2.9 us x 16 = 46 us of the 905 us decode_1 load, ~+0.85% headline
- `*layer=sliding load=verify_32 *` faster: tiled pre-attention 4.2 -> 2.7 us measured twice; shape 29.2 -> ~28.1 us; ~+0.14% headline
- `*layer=sliding load=verify_16 *` flat: kernel 2.8 -> 2.0 us but the shape moved only 22.67 -> 22.74 in c2ebf6ed; routed only if the whole-step gate says <= 0.98
- `*layer=sliding load=decode_* *` flat: master byte for byte (cluster C=16, Triton, sk window)
- `*layer=full load=decode_16 * *layer=full load=decode_32 *` flat: master v2 stream-K (+skF gate) unchanged
- `*layer=full load=verify_* *` flat: paired FA3; tiled expected to be gated off (0.1 us at FV16)

## Risk

Medium risk: a new CUDA module that was judged correct once (8d4771feca9b4e18, 16/16 tests, chunk320 self-check on the judge) and a Triton kernel judged twice. Failure modes are known (no past fault); both are behind per-class gates with master fallback.

## Re-base (why the other named sources are not in any combination)

Ratio = source's paired speedup on the row / master's mean paired speedup (252abb7f, 917a9689); > 1 is faster than master today.
- sliding-cluster-s2-decode-c4-pdl-inproc: its sliding verify 1.69/1.83/2.23x were against sliding-cuda-cluster-dsmem-onelaunch, which had broken verify (28.5/41.5/65.6 us); its log routes every verify to FA3 ('sliding master for ... every verify (FA3)'), so those rows are master's FA3 (re-based 0.80/1.00/0.98). Its decode_32 C=4 is already in master and superseded by the sk window (20.15 vs 17.68 us).
- full-verify-wgmma-s3-warpgroup-per-kvhead-inproc: 'PATH CHOSEN: full verify master FA3 ... (fallback: self-check failed (b8: case 2 rel L2 1.23e+00))'. Its 1.50x verify_16 is FA3 vs a regressed base (244 us); 162.6 vs master 163.0.
- full-verify-wgmma-s5-qfrag-reload: gate kept FA3 (0.57x); its full-decode 'gains' are master's stream-K vs an old base.
- full-decode1-sk-pdl-from-preattn: 'no PDL' chosen (plain 27.74 = best; PDL attribute +0.54 us, trigger +3.84 us); its full decode_32 1.14x is drift on an untouched row (that row moves ~6% between identical runs).
- full-decode1-sk-two-level-fused-preattn: real (gate 1.060x, judge 21.60 vs 22.40 us, re-based 1.04x) but slower than chunk320 on the same row, so it is not a top-row source; combination 2 reuses its prologue instead.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: landed (master eb1b68b); its skF FUSED_PRE was gated off in both runs (B=16 0.988-0.996x, B=32 1.014-1.015x) and still is.

## Winner map (master us from 252abb7f; best non-master source today)

| class | w | master | best other | ratio | floor |
|---|---|---|---|---|---|
| S dec1 | 46 | 11.9 cluster C=16 | none | - | 0.1 |
| F dec1 | 16 | 22.4 two-level sk | chunk320 19.49 | 0.87 | 4.3 |
| S dec16 | 46 | 15.5 Triton | sk-window p8 (in-gate 1.006x) | ~1.0 | 1.6 |
| F dec16 | 16 | 111.1 sk v2 | skF 0.99x in-gate | - | 63.2 |
| S dec32 | 46 | 17.7 sk window | none | - | 3.3 |
| F dec32 | 16 | 227.9 sk v2 | skF 1.015x in-gate | ~0.99 | 126.4 |
| S ver1 | 46 | 13.2 pseudo | none | - | 0.1 |
| F ver1 | 16 | 30.5 FA3 | none (FULLV 0.64x) | - | 4.3 |
| S ver16 | 46 | 22.8 FA3 | tiled pre-attn (kernel -0.8 us) | ~1.0 | 1.9 |
| F ver16 | 16 | 163.0 FA3 paired | none | - | 63.4 |
| S ver32 | 46 | 29.2 FA3 | tiled pre-attn 27.55 | 0.96 | 3.7 |
| F ver32 | 16 | 302.3 FA3 paired | none | - | 126.8 |

## Conjecture

This is the naive stack: the two row wins touch disjoint classes and disjoint kernels, and every shape is timed alone with a cold L2, so the expectation is plain additivity (~+1.0-1.2% headline, 1.502 -> ~1.517-1.520). The interaction it tests is whether chunk320's 0.87x survives on today's master at all: it was measured against FA3 on an older lineage, and master now builds three more CUDA modules and runs more import-time work before the B=1 kernel is captured, so a context/occupancy or L2-residency effect from the extra modules, or a timing shift from the new 2-CTA/SM 250-CTA grid running right after a Triton node, would show as GATE A's prepare ratio holding (~0.87) while the judged full decode_1 row moves less than -2 us. Confirm: GATE A line plus the profile row (splitkv_attn_kernel<320> ~11 us, splitkv_combine_kernel<320,16> ~4.5 us, shape ~19.5 us). Refute: the row stays ~22.4 with the c320 kernels in the profile (the old run's gain was base-specific), or GATE A declines. Secondary: the tiled kernel's sliding verify_32 win (measured on 1.386-era master) should reproduce as _pre_attention_tiled_kernel ~2.7 us in the verify_32 sliding profile.
