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
status: running
review_at: 2026-10-25T162247Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T162247Z
base_sha: 7b45d97a3cf25fe91a77f16e8cb649cf1d2c3812
base_blob: 4bb57fd292a7f0fd5a80bd88dc773008b490bf8c
branch: hyp/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t
worktree: .worktrees/muse_decode_layer/combo-measured-rows-plus-rega-align2-large-t
mode: null
restack_of: null
stale_base: null
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

