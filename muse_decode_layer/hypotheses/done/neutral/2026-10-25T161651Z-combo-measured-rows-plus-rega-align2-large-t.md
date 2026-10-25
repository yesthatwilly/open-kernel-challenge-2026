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
review: proceed
status: done
review_at: 2026-10-25T162247Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T202838Z
base_sha: 955dc77c0c4e24a2271f21aa7a6691e2fd1d3086
base_blob: 3fc72e97304e132db432fcc87872be997f842929
branch: hyp/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t
worktree: .worktrees/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T204200Z
impl_commit: d3f01b4424fbe4f3802c90c5b666132bd2c1ff1d
submission_sha256: 71486a3937c60c91670a0c5a0ac9dc7876a21e7f8b22accde5a60f5c9d5faf18
norm_sha256: 21a2a380cebb677a170e239831e033ec42f10e306595c2dc81fca5d56ea8125c
submission_blob: fc88ac07e23933ce3f8951894577f1b70ec2b0ff
lint_warnings: 96
attempt_file: 2026-10-25T204200Z-submit-combo-measured-rows-plus-rega-align2-large-t.md
prepared_at: 2026-10-25T204200Z
job: eaa8cc0f18f64bfb
dispatched_at: 2026-10-25T204209Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.316368461737821
speedup_ci: [1.2966229348117537, 1.3361139886638882]
harvested_at: 2026-10-25T204641Z
best_at_harvest: 1.3142361220457452
promotion: null
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

## Adversarial review

Reviewed 2026-10-25T162247Z by an adversarial subagent (model opus).

**Verdict: proceed.** Safe to run as a scored combo: every new cell is gated in prepare(), the expert-weight copy fits easily, and M1 can be read directly from the profile and its PATH CHOSEN line. Three things must change before the build: (1) decide align2 BEFORE the rega checks, or keep rega on align_par; (2) put back the standalone port's align-equivalence step; (3) make the copy check robust to decode's state, which holds a torch.cuda.Stream. The B128 conjecture rests on a false premise, and M2's join-slack conjecture cannot be read from a ~1.5 us residual.

### Justification
M1 mechanism: plausible but the expected size is overstated. At T=256 decode's incumbent is a stronger Marlin than the one moe beat. moe's _fwd_block_size_m gives bsm 32 at T=256 (64 -> two-smaller), but decode is capped at 16 (marlin-block-m-cap-16), and decode's Marlin pair is 409.2 / 442.7 us (sliding/full, job 933d) against moe's 457 (selector) / 483 us (judged). The 'expected' line takes moe's absolute margin (457 - 402 = 55 us) instead of comparing against decode's own incumbent. If rega costs about what it did in moe (~402 us, with moe's busier routing), sliding verify_32 gains only ~7 us (1.7%, below the 1.02 gate) and full ~40 us. A realistic outcome is therefore B256 = REGA on one or neither layer's routing, or a win smaller than 40-55 us; the selector protects the downside either way. The expert-block floor is ~170 us (32 experts x ~816 MiB NVFP4 at 4.8 TB/s), so there is room in principle. WEAKEST LINK: the Conjecture's first premise is false. It says decode's B128 incumbent 'is a different Marlin (block_m capped at 16, so at T=128 it launches more blocks than moe's)'. But moe's T=128 also runs bsm 16 (SGLang picks 48, two-smaller gives 16), so the B128 Marlin config is identical in both problems. The remaining arguments are flatter, fewer local routes (plausible, but it shrinks both arms) and 'input arrives after a side-stream join' (irrelevant: the selector times the expert block in isolation). With moe measuring rega at 0.80-0.81x Marlin at B128, the prediction 'B128 = REGA' is weakly supported. The run CAN refute it cleanly (the [rega] B128 line with both numbers and local-route counts), which is fine, but the author should expect MARLIN. M2: align2 saves ~1.8 us on a side branch (~26 us) that races the pooler nvjet (~35 us at T=256) plus the pre-expert norm. It is most likely hidden, and even if it is on the critical path the verify_32 row moves ~1.5 us on 607-869 us, far inside the +/-1.5% CI (9-13 us). The proposed read-out (row delta minus the selector's expert-block delta, residual ~1.5 us) therefore cannot be resolved by this run. Either drop M2 or make it testable inside prepare(): time the whole T=256 FFN front as one graph (pooler + norm on main || router + routing + align on side -> join) with align_par vs align2, and print both. That measures the join slack directly. Time and memory: fine. Master's whole check took 216 s wall (prepare 4.5 s). Adding the C1 modules, the rega load_inline (~45-60 s) and ~85 s of prepare phases still lands well under 300 s for prepare and 15 min for the run, and the per-phase deadlines will not bite. The extra NVFP4 bank is w13 512 MiB + w2 256 MiB + scales ~48 MiB = ~0.8 GiB per state. The judge prepares once and clones the state (the 933d log shows a single prepare with cumulative 4.5 s), so the second copy is +0.8 GiB, plus ~0.8 GiB of transient clones in prepare(). That is about 2.4 GiB peak extra on an H200 (141 GB), alongside the existing 512 MB cos_sin table per state: no memory concern. WG_CAP_ROWS 8192 and WG_MAX_BLOCKS 384 cover decode's T=512 bsm-16 align (cap 4576 rows, 286 blocks). Shapes match moe: hidden 4096 into the experts, size_n 4096, 32 local experts. Copy-safety: moe s5's per-call tensor-map encode into a stack-local grid constant, with per-state wg_xs / wg_hs / wg_err and no stored map, is the right design for the judge's clone-and-rotate. Under capture the encode bakes the capturing state's pointers into that graph, which is what the judge does.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1 (landed): decode's T>32 align is _align_parallel on the SIDE stream, consumed after main_stream.wait_stream. moe's rega path never consumed that align: _moe_wgmma calls SGLang moe_align_block_size(local_ids, 16, 32) inline on the main stream. Its outputs are the same capacity (n + 32*15) with sentinel n and block_experts -1 past used, so they should match what rega reads (it is bounded by n_padded and skips -1 blocks). The combo nevertheless dropped the standalone port's step (3), the bit-for-bit check that rega fed align_par equals rega fed moe_align_block_size(local_ids, 16, 32). Restore it; it is cheap.
- M2 align2 vs M1 rega (the moe designer's note, confirmed in moe master): in moe, align2 runs only inside _moe_marlin, never on the wgmma path, so align2 -> rega has never run anywhere. In this combo, kernel_fn's dispatch would feed align2's output to rega at v256 whenever both flags are set. M1's phase (deadline 200 s) runs its canary, three-state copy check and selector BEFORE M2's phase (250 s) sets the align2 flag, so the composed align2 -> rega path is never validated or timed. Fix: run M2's phase first, so M1's checks and selector go through the final dispatch. Or pin _experts_rega's dispatch to _align_parallel whatever the align2 flag says, as moe effectively did. The bit-identity proof covers the composition only if it compares all four buffers in full (sentinel tail and block_experts past used included), as moe's _align2_first_mismatch does; require that.
- marlin-block-m-cap-16 (landed): sets decode's B256 incumbent at bsm 16, where moe's is bsm 32. It is the reason moe's 0.88x will not transfer directly at T=256 (decode's Marlin is ~10-15% faster than moe's). At B128 the incumbent configs are IDENTICAL (both bsm 16), contrary to the Conjecture. The WG_BSM == _block_size_m(T) assert is correct and needed.
- C1's S3 (pre-attention fused kernels, per-class gated): if any S3 class reaches T=128/256, the verify_16/32 rows carry S3 as well as M1. It does not confound M1 attribution, because the profile separates wg_gather / wg_kernel from Marlin and the pre-attention kernels. Read M1 from the kernel rows, not the shape row.
- small-t dense chain / dl3 pieces / w2 atomic T=1 / post-expert-norm-triton-small-t: T<=32 only. They do not interact with M1/M2 except through the L2/TLB-footprint hypothesis in the Conjecture, which the cold-L2 judge makes unlikely.
- full-decode sk ticket kernels and fa3-hdim128-head-pairing-batched: these touch the attention side of full verify_16/32 rows before the MoE. No interaction with the rega launch sequence (rega starts after the router/align join).

### Interference and A/B needs
No companion warranted. M1 is directly attributable: its PATH CHOSEN line gives decode's own rega-vs-Marlin medians per class, and the judged profile shows wg_kernel / wg_gather time replacing Marlin + act_and_mul on the verify_16/32 rows. A null, gain or loss on those rows is therefore readable at kernel level whatever S1-S3 do. C1's sources live on T<=32 rows (and are measured by C1 itself). M2 is not attributable by any one extra run: its effect (~1.5 us at most) is below the per-shape CI. A companion would spend a scored run to measure noise, so the fix is the in-prepare FFN-front join timing described in the justification, or dropping M2. The standalone moe-rega-bucket-gate-port-t128-t512 (queued, p1) is a strict subset of this combo's M1. Once this runs, mark it superseded rather than spend another run on identical M1 code.

### Open questions
- The three-state copy check cannot be 'copied verbatim'. moe's version captures moe's kernel_fn on (A, copy.deepcopy(state) with the ext memoised, judge-model clone C). Decode's state holds a torch.cuda.Stream ('side_stream'), the dense-chain / sk / dl3 load_inline module handles and nn.Module boundaries. A raw copy.deepcopy may raise, which silently turns M1 off as a 'fallback'. Should the check run decode's T=128/256 expert path (side-stream dispatch + _experts_rega) on states built by the judge-model tensor-cloning walk only, with the stream and ext handles shared via memo? Whatever is chosen, the fallback line must print the exception text.
- Does the M1 selector time rega with the align that kernel_fn will actually use (align2 at v256 if M2 is enabled)? Running M2 before M1 answers this.
- Is there one shared clock for all the prepare() deadlines? C1 120/150/180 s, M1 200 s, M2 250 s are all measured from t_prepare0, but master's prepare finishes in ~5 s, so these deadlines only matter if a phase hangs. Is any phase's own internal timeout (e.g. an mbarrier-hang canary) bounded?
- Expected B256 win: compare against decode's own Marlin (409 / 443 us), not moe's 457. If the selector prints REGA only for the full-layer routing's local-route count, is the flag per class (shared by both layers) or per (class, layer)? The spec is per class, so one routing mix decides for both layers. Print the local-route counts that drove it.
- If M2 is kept: will its selector enable align2 even when the side branch is fully hidden behind the pooler, adding a cell that changes nothing measurable? Prefer gating it on the FFN-front join timing.


## Implementation notes

Implemented on current master (ec89382b, which already carries combo-measured-rows-slide-batched-preattn-t8-tail, i.e. the whole "measured rows" half: S1, S2a/S2b, S3 with their per-class gates). Only the not-yet-on-master part was added: M1 (rega bucket gate) for 128 <= T <= 512. Diff:  1 file changed, 1413 insertions(+), 1 deletion(-); one existing line replaced (the `_experts` call in kernel_fn's T > 32 branch became an if/else).

Reconciliations / deviations (all recorded):
- S1/S2/S3: already on master; NOT re-ported. T <= 32 is exactly master.
- M2 (align2) NOT ported. The adversarial review says drop it or make it testable (its ~1.5 us effect is below the per-shape CI and the join slack cannot be read from a residual), and its composition with rega was never validated anywhere. Rega is therefore pinned to master's `_align_parallel` (review confounder 1 resolved by construction).
- Port source: moe commit 40e2e860c4 (s5), not moe's later master (no PDL variant): the CUDA TU, wg_gather, wg_kernel<true|false>, per-CALL tensor-map encode, wg_init, retile (_wg_bank), nvfp4 decode, routing helpers and the tensor-cloning walk are verbatim (module renamed muse_decode_layer_rega_v1, tag "[rega]"). Build flags kept as moe's proven set (-O3 -lineinfo + sm_90a gencode, TORCH_CUDA_ARCH_LIST=9.0a around load_inline, restored) rather than adding -std=c++20, to not perturb a TU that scored; built at import, never raises (None keeps Marlin), prints compile seconds.
- Compiled module is held in the import-time global _WG_EXT (a handle, like master's _DL_EXT/_SK), NOT in the state, so a deepcopy/clone of the state cannot trip on a module object. Everything device-side (wg_xs, wg_hs, wg_err, wg_bank, wg_grid) is per state.
- prepare(): `rg_clones = _rg_clone_weights(weights)` is the first use of the weights (right after t_prepare0). The retile runs from the clones inside _rega_prepare, which runs AFTER _combo_prepare (so after _dl3_finish), start deadline 200 s from t_prepare0, selector skipped past 240 s; the clones are freed after. It never raises; any miss leaves state["rega"] = {128: False, 256: False} and frees the bank/scratch.
- kernel_fn: in the T > DL_MAX_TOKENS branch after main_stream.wait_stream(side_stream), `_rega_class(tokens)` (shape only: 128 for 128..255, 256 for 256..512, None otherwise) and `state["rega"][class]` pick `_experts_rega(xn, tw, local_ids, align, state)` (gather, w13+silu, w2 into cache3 at the route index, master's fused_post_expert_norm_weighted_sum) over master's `_experts`. No host sync, no print, no try/except, no input-sized allocation.
- Hard guard added (not in the spec): M1 is skipped unless state["align_par_ok"], because wg_gemm TORCH_CHECKs block_experts.numel() <= 384 and the 256-expert moe_align_block_size fallback exceeds that for T > ~290 (a raise inside kernel_fn). With _align_parallel the cap is 286 blocks at T = 512. Also asserted in prepare: WG_BSM == MARLIN_BLOCK_SIZE_M_CAP == _block_size_m(T) for every T in 128..512, and expert weight shapes.
- Review fix 2 (align equivalence): every canary case also runs rega on moe_align_block_size(local_ids, 16, 32) and requires rel-L2 <= 1e-3 against rega fed _align_parallel's align.
- Review fix 3 (copy check): a raw deepcopy of decode's state is NOT attempted (it holds a cuda Stream, Modules). The copy check uses the rega-relevant keys only: state A (prepared), B (copy.deepcopy of those keys), C (recursive tensor-cloning walk after a 2 MB pad); per T in (128, 256, 257) one graph per state, replays A,B,C,A,B per routing (random, skewed-4hot, all-one-expert, mixed) with inputs refilled, each vs an fp32-dequant reference, bound max(2x Marlin's own error, 1e-2), plus the launch marker in each state's own wg_err. Exceptions are printed with their text and turn M1 off; C must exist.
- M1 correctness check: canary at T = 128 forcing N 16/32/64/128 (one expert with 16/32/64/128 routes, bucket asserted from the align), poisoned scratch/cache3, first non-finite tensor named (xs / hs / cache3 / out), error word and launch marker read back, every launch announced before and synced+confirmed after.
- M1 selector (decode's own flags, nothing inherited from moe): per class T = 128 and T = 256, master's whole expert block (dispatch + Marlin + act_and_mul + Marlin + post-norm) vs rega's (dispatch + gather + w13+silu + w2 + post-norm), cold-L2 graphs via master's _cb_time (256 MB flush, 41 interleaved reps, empty graph subtracted), on (a) judge-like routing (real router_hi/lo and bias through FusedMuseSparkRouting on synthetic randn inputs) and (b) zipf a = 0.3; enabled only if rega/Marlin <= 0.98 on BOTH. One flag per class (shared by both layers); the flag set is the printed winner. Prints "[rega] PATH CHOSEN B128|B256 = REGA|MARLIN [rega X / Y us vs marlin X / Y us on soft / skew, local routes N / N]" and "PATH CHOSEN: M1 <class>: ...".
- Unverifiable here (no GPU): nothing was run. Biggest risks: the ~1400-line TU compile at import (+45-60 s; failure falls back silently to Marlin), and the selector keeping B256/B128 on Marlin (the review expects at most a small B256 gain and MARLIN at B128).

## Rules check

- [x] CUDA-graph capturable: git diff shows kernel_fn's new branch calls only _rega_class (shape) and _experts_rega (launches + torch.empty of shape routes x 4096, fused norm); .item()/.cpu()/synchronize appear only in prepare()-time helpers (_rg_case, _rg_copy_check, _rega_prepare), none reached from kernel_fn; the only added control flow reads config tokens and state["rega"] flags.
- [x] Nothing derived from data cached across calls: the diff adds state keys wg_xs/wg_hs/wg_err/wg_grid/wg_bank/rega only; wg_xs/wg_hs are scratch rewritten every call, tensor maps are encoded per call into a stack-local grid constant.
- [x] Weights: wg_bank is a lossless retile of the same NVFP4 bytes (_wg_tile_expert asserts exact round trip), global scales stay fp32 with the exact x128 dequant bias; dequant is in registers inside wg_kernel; the BF16 dense weights are untouched; only prepare() builds it.
- [x] Activations BF16 (xs, hs, cache3 bf16 as SGLang's Marlin path); KV cache and q paths untouched by this diff.
- [x] n/a: attention untouched (every key still attended; the diff touches only the expert block at T >= 128).
- [x] No timing patches, harness frames/I-O/secrets, module edits, reference import, subprocesses or network: diff adds import copy / gc only; load_inline is the same mechanism master uses; no new subprocess/os.fork/threads.
- [x] All work inside the timed call; no output caching or replay: rega runs gather + two GEMMs + norm per call from that call's inputs.
- [x] Exact fallbacks: _WG_EXT None, align_par_ok False, clone failure, deadline, canary/copy/selector miss each leave state["rega"] False => master's _experts, launch for launch.
- [x] Per-state buffers: no device pointer, tensor map or counter lives in a module global (audit by reading the diff: module globals added are constants, _WG_EXT handle and _RG_SELECT floats only).
- [x] Interpreter checks: py_compile clean, ast undefined-name scan empty, __pycache__ removed.

okc_lint output:
```
OK with warnings: .worktrees/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t/muse_decode_layer/submission.py
```
(plus ~110 prepare-only host-sync warnings, all pre-existing style or in new prepare()-only code, and 2 uint8 dtype warnings for the retile scratch: NVFP4 bytes, same as moe.)

## Pre-flight audit

Audited 2026-10-25 by an opus pre-flight auditor (by reading, no GPU). **Verdict: dispatch** (no fixes applied).

- Import-time build: `_wg_build` catches every exception, and the module-level block has its own try/except, so a build failure leaves `_WG_EXT = None` and the module importable. `_rega_prepare` then prints `PATH CHOSEN ... MARLIN (fallback: module ... unavailable)` for both classes. The compile (~45-60 s) counts against the 15 min run limit only, not against prepare()'s 300 s. Moving it into prepare would eat that budget, so it stays at import, as master's modules do.
- Fallbacks: every gate (deadline, clone, align_par_ok, ext, BSM assert, weight shapes, smem and attribute/register checks, bucket canary, align-equivalence, A/B/C copy check, selector) raises `_RgSkip` inside one try. Any miss sets both flags False, frees the bank and scratch, and prints the reason per class. prepare() wraps the phase in a second try.
- Per-state: wg_xs, wg_hs, wg_err, wg_bank and wg_grid live in the state. The tensor maps are encoded per call into a stack-local `__grid_constant__`. The only module globals are the `_WG_EXT` handle, `g_encode_pfn` (a host driver entry point) and `_RG_SELECT`, which holds selector floats only.
- M1 gate: it requires `align_par_ok`, and the kernel_fn branch is reachable only for 128 <= T <= 512. With `_align_parallel` at bsm 16 the cap is 286 blocks at T = 512, within MAXB 384 and WG_CAP_ROWS 8192.
- CUDA TU: byte-identical to moe 40e2e860c4 (`_WG_CPP`, `_WG_SRC_HEAD` and `_WG_SRC_BODY` compared). It has sm_90a guards, error word 2, bounded mbarrier waits, plain bulk copies, `-lineinfo`, and prints its arch flags. There is no subprocess.
- Shapes: decode's w13 [32, 8192, 2048] and w2 [32, 4096, 2048] match the retile's size checks (q13 18,874,368 B and q2 9,437,184 B per expert = tiles x k/64 x 4608). The sentinel n and the -1 tail of `_align_parallel` match what `wg_gather` and `wg_kernel` skip. The remote-route rows of cache3 are masked by `slot_expert_ids`, as in master's Marlin path.
- kernel_fn: no sync, print or try. The branch depends only on config and this state's flags, and the only allocation is cache3, sized by shape.
- Deviations accepted: M2 (align2) was dropped, which the review allowed. The build keeps moe's proven flags without `-std=c++20`.
- Lint: okc_lint OK with warnings (uint8 retile/pad dtype notes, prepare-only syncs); py_compile clean; __pycache__ removed.

Residual risk: a device fault (sticky illegal instruction) in the prepare canary would fail the whole run instead of falling back. The selector may keep both classes on Marlin, which would make this a neutral run.

## Result

- job `eaa8cc0f18f64bfb` (submit), status done, passed True, tests 13/13
- speedup 1.3164x, 95% CI [1.2966, 1.3361], score 17183.866567372417 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.3142361220457452, "ci": [1.294522580215059, 1.3339496638764312], "job": "a903b9d004d2452d"}; base result: muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-plus-rega-align2-large-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-plus-rega-align2-large-t.patch`; branch `hyp/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 118.48 | 117.26 | 1.0121 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 124.31 | 122.83 | 1.0063 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 210.54 | 201.05 | 1.0095 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 290.53 | 284.65 | 0.9999 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 239.48 | 239.27 | 1.0041 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 446.42 | 431.18 | 1.0131 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 181.08 | 174.79 | 1.0269 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 189.51 | 187.87 | 1.0125 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 414.69 | 425.85 | 0.9674 | lost |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 593.41 | 579.72 | 0.9494 | lost |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 599.76 | 591.47 | 1.0094 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 866.62 | 901.01 | 0.9897 | flat |

lost on sliding verify_16; full verify_16. Expected hit: 1/7. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

