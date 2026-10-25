---
problem: muse_decode_layer
slug: combo-measured-rows-plus-rega-align2-large-t
created: 2026-10-25T161651Z
base: master
area: "combination: C1 measured stack + w13/w2 rega and align2 at 128<=T<=512"
change: >
  COMBINATION (measured stack + cross-problem MoE at large T): C1's three decode-measured sources
  (S1 Triton sliding batched 2..32 from sliding-fused-batched-decode-port 3ddcc4f2a8 /
  sliding-batched-triton-b16-band 1cabea713e; S2 R variant 0 at 2 <= T <= 32 + unpool NF=1 at 2 <=
  T <= 8 from stack-route-topk-pdl-unpool-nf1 2fe4d35e13; S3 _pre_attention_kernel +
  _paired_norm_gate_kernel from fused-preattention-paired-pad-restack 963bcf0267, per-class gated
  exactly as C1 specifies) PLUS two muse_moe_w4a16 master sources for T > 32, on master
  6773ec5cf5. SOURCE M1 rega bucket gate (moe commit 40e2e860c4, hypothesis
  cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, scored 4e4a0e26c85944f2 1.3727x; its
  selector chose B256 = WGMMA, B128 = MARLIN on moe's routing; same port as queued
  moe-rega-bucket-gate-port-t128-t512): copy verbatim the WG_* constants, the CUDA TU (wg_gather,
  wg_kernel<true> w13+silu into wg_hs, wg_kernel<false> w2 into cache3 at the route index,
  per-CALL tensor-map encode into a stack-local grid constant, driver entry point resolved once in
  prepare, no stored map or pointer record), _wg_attach_scratch (per-state wg_xs / wg_hs /
  wg_err), the bank retile, the per-bucket canary forcing N 16/32/64/128 and the three-state
  copy-safety check; module renamed muse_decode_layer_rega_v1. Decode adaptation: the FIRST
  statement of prepare() clones weights['w13','w13_sf','w13_gs','w2','w2_sf','w2_gs'] (memory
  only; master's _marlin_bank may repack in place), and the retile runs from the clones after
  master's setup, then the clones are freed. In kernel_fn's T > DL_MAX_TOKENS branch, after
  main_stream.wait_stream(side_stream), class B128 (128 <= T <= 255) and class B256 (256 <= T <=
  512) each call _experts_rega(xn, tw, local_ids, align, state) on their own flag, then master's
  fused_post_expert_norm_weighted_sum finish. Decode's align at T >= 128 is bsm 16 under
  MARLIN_BLOCK_SIZE_M_CAP, which equals WG_BSM: assert both in prepare. SOURCE M2 align2 (moe
  commit a0e52640, hypothesis align-two-phase-port-from17; moe chose it only at v256: 6.46 -> 4.70
  us at bsm 32, master's align elsewhere): _align2_count_kernel + _align2_place_kernel replacing
  _align_parallel on the side stream for class v256 only (and v128 only if its own selector says
  so), re-proved bit-identical to _align_parallel at decode's bsm 16 (moe proved it at bsm 32) on
  skewed, random, all-one-expert and no-local routings with local_start in {0, 96, 224}. M1 check:
  rel-L2 of the post-combine output vs an fp32-dequant reference with master's Marlin error
  printed beside it, bound max(2x Marlin's, 1e-2); name the first non-finite tensor. M1 selector
  (decode's own, do not inherit moe's flags): per class at T=128 and T=256, decode's whole expert
  block (align + Marlin w13 + act_and_mul + Marlin w2 + post-norm) vs (align + gather + w13silu +
  w2 + post-norm) on (a) judge-like routing from the real router weights and bias through
  FusedMuseSparkRouting on synthetic rmsnormed inputs and (b) zipf a=0.3 skew; enable only if <=
  0.98x on both. M2 selector: the whole side-stream branch (router hi/lo + routing + align) at
  T=256 and T=128. Print '[rega] PATH CHOSEN B128|B256 = REGA|MARLIN [rega X us vs marlin Y us on
  soft / skew, local routes N]' and '[align2] PATH CHOSEN v128|v256 = ...'. T <= 32 is exactly C1.
  HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no
  prints, no try/except, no allocation sized by input values (shape/config only), no
  data-dependent Python control flow and no per-call mutable Python state; gates read config/shape
  and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4
  stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key
  attended; no child processes. The judge prepares TWO state copies and rotates them: every device
  pointer, counter/ticket array, workspace and TMA map lives in the state, never in a module
  global (this includes the Triton sliding kernel's split counters: master's module-global
  _COUNTERS dict must NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per
  state and slice it). Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed
  load_inline module built at import like master's, with an in-process canary (synchronize +
  'launching'/'ok' prints, flush=True) after each first launch; on any miss a 'PATH CHOSEN ...
  (fallback: <reason>)' line keeps master for that cell. GATE PER SOURCE PER CLASS: each source
  path has its own prepare-time gate = (a) correctness check (bound stated per source below) on
  two alternating states with refilled inputs under captured-graph replay, then (b) a cold-L2
  CUDA-graph selector (256 MB flush before each replay, >= 41 interleaved reps, empty-graph
  baseline subtracted) that enables the cell only when new/master <= 0.98 (i.e. master/new >=
  1.02) and asserts the enabled flag equals the printed winner. Print exactly one 'PATH CHOSEN:
  <source> <class>: <winner> [new X us vs master Y us]' line per source per class. Every added
  prepare() phase runs AFTER master's own setup (after _dl3_finish), each with its own start
  deadline measured from prepare()'s t_prepare0 (skip the phase with a fallback line if the
  deadline has passed) and a print of its own wall seconds plus the cumulative prepare wall time;
  master's own internal deadlines and its _sk_prepare budget do not move. Print compile seconds
  per load_inline module. First measurement is an okc check.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "M1 B256: Marlin 409.2 / 442.7 us (sliding/full, 933d) vs moe's 402-vs-457 margin -> ~-40..-55 us per shape (~7% of the verify_32 load); M2 -1..-2 us if the side branch is critical"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "M1 B128 only if decode's own selector flips it (moe kept Marlin): Marlin 262.8 / 307.9 us; else flat"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: S1 + S2 as in C1
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: S1 + S2 as in C1
  - shapes: "*load=verify_1 *"
    direction: faster
    why: S2 + S3 as in C1
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: S3 c1 + S2a as in C1
  - shapes: "*load=decode_1 *"
    direction: flat
    why: untouched; drift control
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-measured-rows-plus-rega-align2-large-t

## Change

COMBINATION (measured stack + cross-problem MoE at large T): C1's three decode-measured sources (S1 Triton sliding batched 2..32 from sliding-fused-batched-decode-port 3ddcc4f2a8 / sliding-batched-triton-b16-band 1cabea713e; S2 R variant 0 at 2 <= T <= 32 + unpool NF=1 at 2 <= T <= 8 from stack-route-topk-pdl-unpool-nf1 2fe4d35e13; S3 _pre_attention_kernel + _paired_norm_gate_kernel from fused-preattention-paired-pad-restack 963bcf0267, per-class gated exactly as C1 specifies) PLUS two muse_moe_w4a16 master sources for T > 32, on master 6773ec5cf5. SOURCE M1 rega bucket gate (moe commit 40e2e860c4, hypothesis cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, scored 4e4a0e26c85944f2 1.3727x; its selector chose B256 = WGMMA, B128 = MARLIN on moe's routing; same port as queued moe-rega-bucket-gate-port-t128-t512): copy verbatim the WG_* constants, the CUDA TU (wg_gather, wg_kernel<true> w13+silu into wg_hs, wg_kernel<false> w2 into cache3 at the route index, per-CALL tensor-map encode into a stack-local grid constant, driver entry point resolved once in prepare, no stored map or pointer record), _wg_attach_scratch (per-state wg_xs / wg_hs / wg_err), the bank retile, the per-bucket canary forcing N 16/32/64/128 and the three-state copy-safety check; module renamed muse_decode_layer_rega_v1. Decode adaptation: the FIRST statement of prepare() clones weights['w13','w13_sf','w13_gs','w2','w2_sf','w2_gs'] (memory only; master's _marlin_bank may repack in place), and the retile runs from the clones after master's setup, then the clones are freed. In kernel_fn's T > DL_MAX_TOKENS branch, after main_stream.wait_stream(side_stream), class B128 (128 <= T <= 255) and class B256 (256 <= T <= 512) each call _experts_rega(xn, tw, local_ids, align, state) on their own flag, then master's fused_post_expert_norm_weighted_sum finish. Decode's align at T >= 128 is bsm 16 under MARLIN_BLOCK_SIZE_M_CAP, which equals WG_BSM: assert both in prepare. SOURCE M2 align2 (moe commit a0e52640, hypothesis align-two-phase-port-from17; moe chose it only at v256: 6.46 -> 4.70 us at bsm 32, master's align elsewhere): _align2_count_kernel + _align2_place_kernel replacing _align_parallel on the side stream for class v256 only (and v128 only if its own selector says so), re-proved bit-identical to _align_parallel at decode's bsm 16 (moe proved it at bsm 32) on skewed, random, all-one-expert and no-local routings with local_start in {0, 96, 224}. M1 check: rel-L2 of the post-combine output vs an fp32-dequant reference with master's Marlin error printed beside it, bound max(2x Marlin's, 1e-2); name the first non-finite tensor. M1 selector (decode's own, do not inherit moe's flags): per class at T=128 and T=256, decode's whole expert block (align + Marlin w13 + act_and_mul + Marlin w2 + post-norm) vs (align + gather + w13silu + w2 + post-norm) on (a) judge-like routing from the real router weights and bias through FusedMuseSparkRouting on synthetic rmsnormed inputs and (b) zipf a=0.3 skew; enable only if <= 0.98x on both. M2 selector: the whole side-stream branch (router hi/lo + routing + align) at T=256 and T=128. Print '[rega] PATH CHOSEN B128|B256 = REGA|MARLIN [rega X us vs marlin Y us on soft / skew, local routes N]' and '[align2] PATH CHOSEN v128|v256 = ...'. T <= 32 is exactly C1. HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, counter/ticket array, workspace and TMA map lives in the state, never in a module global (this includes the Triton sliding kernel's split counters: master's module-global _COUNTERS dict must NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per state and slice it). Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed load_inline module built at import like master's, with an in-process canary (synchronize + 'launching'/'ok' prints, flush=True) after each first launch; on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line keeps master for that cell. GATE PER SOURCE PER CLASS: each source path has its own prepare-time gate = (a) correctness check (bound stated per source below) on two alternating states with refilled inputs under captured-graph replay, then (b) a cold-L2 CUDA-graph selector (256 MB flush before each replay, >= 41 interleaved reps, empty-graph baseline subtracted) that enables the cell only when new/master <= 0.98 (i.e. master/new >= 1.02) and asserts the enabled flag equals the printed winner. Print exactly one 'PATH CHOSEN: <source> <class>: <winner> [new X us vs master Y us]' line per source per class. Every added prepare() phase runs AFTER master's own setup (after _dl3_finish), each with its own start deadline measured from prepare()'s t_prepare0 (skip the phase with a fallback line if the deadline has passed) and a print of its own wall seconds plus the cumulative prepare wall time; master's own internal deadlines and its _sk_prepare budget do not move. Print compile seconds per load_inline module. First measurement is an okc check.

## Expected

- `*load=verify_32*` faster: M1 B256: Marlin 409.2 / 442.7 us (sliding/full, 933d) vs moe's 402-vs-457 margin -> ~-40..-55 us per shape (~7% of the verify_32 load); M2 -1..-2 us if the side branch is critical
- `*load=verify_16*` faster: M1 B128 only if decode's own selector flips it (moe kept Marlin): Marlin 262.8 / 307.9 us; else flat
- `*layer=sliding load=decode_16*` faster: S1 + S2 as in C1
- `*layer=sliding load=decode_32*` faster: S1 + S2 as in C1
- `*load=verify_1 *` faster: S2 + S3 as in C1
- `*layer=full load=decode_16*` faster: S3 c1 + S2a as in C1
- `*load=decode_1 *` flat: untouched; drift control

## Risk


C1 in this file means the queued combination combo-measured-rows-slide-batched-preattn-t8-tail (same mint, 2026-10-25T161651Z): its sources S1/S2/S3, their gates and checks are reused verbatim here.

The highest-ceiling candidate: C1's expected ~+1.5..+2% plus M1's ~+1.1% (verify_32 load) and maybe B128, about +3% headline (~1.35). Rows split cleanly for attribution against C1 run the same day: T <= 32 rows should match C1, T >= 128 rows carry M1/M2 (S3's verify_16/32 cells were flat when measured). Prepare phases (after _dl3_finish): C1's S2/S1/S3 ~40 s (deadlines 120/150/180 s as in C1); M1 retile from clones ~5 s, canary ~3 s, three-state copy check ~15 s, selector 2 classes x 2 routings ~12 s (start deadline 200 s, its timing skipped past 240 s); M2 equivalence at bsm 16 ~3 s + selector ~3 s (deadline 250 s); total ~85 s. Import compile: +1 module (rega, ~45-60 s; moe's whole rega job ran 124 s). Memory: one extra NVFP4 bank (~0.8 GiB) per state plus the transient clones.

## Conjecture

moe's B128 = MARLIN verdict will not transfer: decode's incumbent is a different Marlin (block_m capped at 16, so at T=128 it launches more blocks than moe's), its EP8 'dist soft' routing gives fewer and flatter local routes than moe's busiest-GPU routing, and decode's rega input arrives after a side-stream join rather than at the head of the graph. The interaction tested is a selector re-deciding in a new neighbourhood: prediction B128 = REGA in decode. Second interaction: at T=256 the side-stream branch (router hi/lo GEMMs 2 x splitK + splitKreduce + fp32 add + FusedMuseSparkRouting + align, ~26 us at full verify_32) races the pooler GEMM + pre-expert norm on the main stream; M2 only pays if that branch is the critical path into the join, and M1's persistent 132-CTA wgmma kernel cannot start until the join, so M2's gain is capped at the join slack. Confirm/refute: the '[rega] PATH CHOSEN B128' line and its local-route counts; the verify_16 row and its Marlin vs wg_kernel rows; for M2, the verify_32 row delta minus the selector's expert-block delta (a residual of ~1.5 us means the side branch was critical; ~0 means hidden). Third: if S1-S3 change the T <= 32 rows differently here than in C1, the extra bank and scratch (L2/TLB footprint) are interacting with the small-T path.
