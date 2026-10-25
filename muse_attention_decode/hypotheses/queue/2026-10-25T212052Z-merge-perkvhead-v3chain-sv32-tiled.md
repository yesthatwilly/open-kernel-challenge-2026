---
problem: muse_attention_decode
slug: merge-perkvhead-v3chain-sv32-tiled
created: 2026-10-25T212052Z
base: master
area: full decode_1 split-KV launch chain (merge onto per-kvhead master, with SV32 tiled pre-attention)
change: >
  DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch
  hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 =
  per-kvhead; job a78f29aeab9b47cb, 1.5347: sliding decode_1 11.05 us, sliding verify_32 28.02 us,
  but full decode_1 19.71 us because it carries c320 V1 = pre-attn + c320 attn + c320 combine,
  three nodes). Replace ONLY that full decode_1 piece with the chain from
  combo-b1-chain-c320-fused-pre-pdl (branch
  hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl, job 8713665f, the current best:
  full decode_1 17.39 us = c320_attn_kernel<320,true> 11.4 + c320_combine_kernel<320,16> 4.2, i.e.
  V3 = FUSED pre-attention prologue in kernel A + combine, with V4 = V3 + PDL A->combine also
  built). Port combo-b1's muse_full_decode_c320 module source (variants V0..V4, FUSED and PDL host
  entry points as separate functions, no mutable toggle) and its host code verbatim: GATE A
  correctness for every variant incl. the new k/v cache row comparison against
  _pre_attention_kernel and the two-state alternating replay at 80000 and 4097 keys, the
  five-variant interleaved pass (41 reps) and the head-to-head re-time of the pass-1 best against
  V0 and V1 (31 reps), routing the best only if best/V0 <= 0.98, and its 'PATH CHOSEN: full
  decode_1 [chain]: V<k>' line. Delete the merge branch's V1-only c320 module and host code so
  exactly ONE c320 module is built (combo-b1's is a superset; the import clock is 222 s across 4
  modules on master and must not gain a 6th build). Keep per-kvhead's sliding decode_1 cluster
  path (2 clusters x C=16) and the SV32 tiled pre-attention with its GATE B exactness and timing
  gate exactly as on the merge branch. Every other class is master byte for byte. Phase order:
  master's own setup first, then the c320 chain (own deadline 120 s), then SV32 tiled (own
  deadline), each falling back to master with one PATH CHOSEN line on failure or overrun, and the
  480 s import start rule kept. No new CUDA: every kernel here already ran on the judge (8713665f,
  a78f29ae). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation
  sized by input values (shape/config only), no data-dependent Python control flow, no prints, no
  try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8
  E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no
  timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may
  hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting
  int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo,
  in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM
  and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2
  graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path
  WINS by the stated margin; PATH CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "vs the merge run 19.71 us (c320 V1): V3 measured 17.39 us in 8713665f (0.791 of master two-level 22.4); about -2.3 us at ~0.33%/us"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "vs the best 8713665f (12.1 us, pre-per-kvhead base): per-kvhead measured 11.05 us in a78f29ae"
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "vs the best 29.0 us: SV32 tiled pre-attention measured 28.02 us in a78f29ae (0.964 of master)"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_1 *"
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "merge-perkvhead-c320-sv32-tiled: that merge ported c320 V1 (three nodes, 19.71 us, lost full decode_1 to the best's 17.39) and this one ports combo-b1-chain-c320-fused-pre-pdl's V3/V4 chain (fused pre-attention prologue + PDL) instead, keeping per-kvhead and SV32 unchanged"
review: required
status: queued
---

# merge-perkvhead-v3chain-sv32-tiled

## Change

DIAGNOSIS action 3(i), the PENDING_USER hand-merge. Start from branch hyp/muse_attention_decode/merge-perkvhead-c320-sv32-tiled (built on master 5d528bf3 = per-kvhead; job a78f29aeab9b47cb, 1.5347: sliding decode_1 11.05 us, sliding verify_32 28.02 us, but full decode_1 19.71 us because it carries c320 V1 = pre-attn + c320 attn + c320 combine, three nodes). Replace ONLY that full decode_1 piece with the chain from combo-b1-chain-c320-fused-pre-pdl (branch hyp/muse_attention_decode/combo-b1-chain-c320-fused-pre-pdl, job 8713665f, the current best: full decode_1 17.39 us = c320_attn_kernel<320,true> 11.4 + c320_combine_kernel<320,16> 4.2, i.e. V3 = FUSED pre-attention prologue in kernel A + combine, with V4 = V3 + PDL A->combine also built). Port combo-b1's muse_full_decode_c320 module source (variants V0..V4, FUSED and PDL host entry points as separate functions, no mutable toggle) and its host code verbatim: GATE A correctness for every variant incl. the new k/v cache row comparison against _pre_attention_kernel and the two-state alternating replay at 80000 and 4097 keys, the five-variant interleaved pass (41 reps) and the head-to-head re-time of the pass-1 best against V0 and V1 (31 reps), routing the best only if best/V0 <= 0.98, and its 'PATH CHOSEN: full decode_1 [chain]: V<k>' line. Delete the merge branch's V1-only c320 module and host code so exactly ONE c320 module is built (combo-b1's is a superset; the import clock is 222 s across 4 modules on master and must not gain a 6th build). Keep per-kvhead's sliding decode_1 cluster path (2 clusters x C=16) and the SV32 tiled pre-attention with its GATE B exactness and timing gate exactly as on the merge branch. Every other class is master byte for byte. Phase order: master's own setup first, then the c320 chain (own deadline 120 s), then SV32 tiled (own deadline), each falling back to master with one PATH CHOSEN line on failure or overrun, and the 480 s import start rule kept. No new CUDA: every kernel here already ran on the judge (8713665f, a78f29ae). RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine). BUILD: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20 whenever extra_cuda_cflags are overridden, -lineinfo, in-process canary, sync + print after each self-check launch, print regs/local bytes/CTAs per SM and refuse a variant with local bytes > 0. Gates compare printed interleaved cold-L2 graph-replay timings (256 MB flush, empty graph subtracted) and route only when the new path WINS by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_1 *` faster: vs the merge run 19.71 us (c320 V1): V3 measured 17.39 us in 8713665f (0.791 of master two-level 22.4); about -2.3 us at ~0.33%/us
- `*layer=sliding load=decode_1 *` faster: vs the best 8713665f (12.1 us, pre-per-kvhead base): per-kvhead measured 11.05 us in a78f29ae
- `*layer=sliding load=verify_32 *` faster: vs the best 29.0 us: SV32 tiled pre-attention measured 28.02 us in a78f29ae (0.964 of master)
- `*layer=full load=decode_16 *` flat: untouched
- `*load=verify_1 *` flat: untouched

## Risk

Expected headline about 1.545 (merge 1.5347 plus ~-2.3 us on full decode_1), i.e. +1.0-1.3% over the best 1.5295: at or just under the 1.24% one-run margin, so it is the base that sliding-verify1-pseudo-fused-preattn-rope stacks on (depends_on) for the scored submit. Risk is low: every kernel already ran on the judge; the risk is the 3-way hand merge and the import clock (one c320 module only).
