---
problem: muse_decode_layer
slug: qkv-o-swapab16-t2-t16-no-fork
created: 2026-10-25T110701Z
base: master
area: qkv GEMM + o GEMM (2 <= T <= 16)
change: >
  Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend:
  qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sites in muse_decode_layer
  kernel_fn: qkv = _gemm(data['attn_input'], dense['qkv_gate']) (n 2304, k 8192) and attn_output =
  _gemm(o, dense['o']) (n 8192, k 1024), gated on config tokens 2 <= T <= 16 (scored: verify_1 T=8
  and decode_16 T=16, both layer types; tests at batch 8/9 too). T=1 is left to the in-flight
  dense-chain-s3 (which rewrites the T=1 qkv/o side) and T >= 17 keeps cuBLAS (that band is the
  held prefork-qkv-o-swapab-t16-t32, whose side-stream-fork premise is stale). Copy verbatim from
  muse_bf16_proj/submission.py on master: _bf16_swapab_kernel (do_not_specialize M, ticket split-K
  reduce), SWAPAB_BLOCK_W=64, SWAPAB_NUM_WARPS=4, SWAPAB16_BLOCK_T=SWAPAB16_WS_M=16, the qkv_gate
  row (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages) and the o row (8192, 1024, SPLIT_K 1, BLOCK_K
  256, 4 stages) of SWAPAB16_CFG, SWAPAB_FALLBACK_STAGES=3 with the OutOfResources import guard,
  and _swapab16_forward. For attribution give each call its own kernel name in the profile: two
  @triton.jit copies (or thin wrappers over a shared @triton.jit body) named _swapab_qkv_kernel
  and _swapab_o_kernel. prepare(): per STATE (the judge prepares two state copies and rotates
  them) allocate the qkv fp32 split-K workspace [16, 2304]-shaped as bf16 does and a zeroed int32
  ticket vector, store them in state (never module globals; o has SPLIT_K 1 and needs none), warm
  each kernel at every T in 2..16 so the timed call never compiles, print '[swapab16-dl]
  layer=qkv|o T=.. stages=.. n_regs=.. wgmma=.. ON|OFF', and self-check each layer against
  torch.mm on random bf16 inputs at T = 2, 8, 9, 16 with rel L2 <= 1e-2, then once more after
  alternating calls on state 0 and state 1 with refilled inputs (ticket reset check). Any failure
  or OutOfResources at both stage counts turns that layer off (cuBLAS _gemm kept). kernel_fn: a
  branch on cfg tokens and the state flag only; nothing else changes (attention, boundaries, dense
  chain P/R, Marlin, unpooler untouched). WHY NOW: LESSONS records that decode's earlier
  bf16-shape-gated-triton-gemms-small-t won kernel sums at T=8/16 yet lost 2-4.5% on those shapes,
  'possibly interference with the side-stream router overlap (unverified)'.
  small-t-launch-minimal-dense-chain-pdl has since removed the router side stream at T <= 32, so
  this run is also the direct test of that open lesson: per-layer kernel rows faster AND shapes
  faster confirms it; rows faster but shapes flat/slower refutes it.
expected:
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: cuBLAS qkv nvjet_sm90_tst_128x8_64x12_4x1 splitK 13.5 us + splitKreduce (~1.7) and o rows (nvjet 64x8 ~7 us) in profile 3fd3fa1f; bf16 m16 swap-AB gave qkv -18%, o -10-19% at this m -> ~3-4 us of 191/202 us, ~2%; read the _swapab_qkv_kernel / _swapab_o_kernel rows against the cuBLAS rows they replace"
  - shapes: "*load=decode_16*"
    direction: faster
    why: "T=16: nvjet 64x16 2x1 (26.2 us over 2 calls) + 64x16 4x1 splitK 12.2 us in profile 3fd3fa1f; same per-kernel saving, ~3-5 us of 205/309 us (sliding weight 46)"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate (dense-chain-s3 territory)
  - shapes: "*load=decode_32*"
    direction: flat
    why: T=32 outside the gate
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 outside the gate
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 outside the gate
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T111209Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T111210Z
base_sha: 54d3a9ce8a155b629ac50f80222a1bf0dec23a79
base_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
branch: hyp/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork
worktree: .worktrees/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T111944Z
impl_commit: 3d76888397175b1738de07494f46f8a23c1124ae
submission_sha256: 5e6ac7de07ee52f358844299a63c7f903e4e68a21e91353d68d004228e6d2038
norm_sha256: e8a071fda51b137637fe6c923240d329818d7883d7dcd5258504670ebec58af4
submission_blob: 188072e2b5c7c7dd0b9b98d87989424d120d5b77
lint_warnings: 66
attempt_file: 2026-10-25T111944Z-submit-qkv-o-swapab16-t2-t16-no-fork.md
prepared_at: 2026-10-25T111944Z
job: b0057d2e6cef4d5a
dispatched_at: 2026-10-25T111956Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3055763257962147
speedup_ci: [1.2859926809092714, 1.3251599706831578]
harvested_at: 2026-10-25T115710Z
best_at_harvest: 1.2725637226942819
promotion: null
---

# qkv-o-swapab16-t2-t16-no-fork

## Change

Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend: qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sites in muse_decode_layer kernel_fn: qkv = _gemm(data['attn_input'], dense['qkv_gate']) (n 2304, k 8192) and attn_output = _gemm(o, dense['o']) (n 8192, k 1024), gated on config tokens 2 <= T <= 16 (scored: verify_1 T=8 and decode_16 T=16, both layer types; tests at batch 8/9 too). T=1 is left to the in-flight dense-chain-s3 (which rewrites the T=1 qkv/o side) and T >= 17 keeps cuBLAS (that band is the held prefork-qkv-o-swapab-t16-t32, whose side-stream-fork premise is stale). Copy verbatim from muse_bf16_proj/submission.py on master: _bf16_swapab_kernel (do_not_specialize M, ticket split-K reduce), SWAPAB_BLOCK_W=64, SWAPAB_NUM_WARPS=4, SWAPAB16_BLOCK_T=SWAPAB16_WS_M=16, the qkv_gate row (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages) and the o row (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages) of SWAPAB16_CFG, SWAPAB_FALLBACK_STAGES=3 with the OutOfResources import guard, and _swapab16_forward. For attribution give each call its own kernel name in the profile: two @triton.jit copies (or thin wrappers over a shared @triton.jit body) named _swapab_qkv_kernel and _swapab_o_kernel. prepare(): per STATE (the judge prepares two state copies and rotates them) allocate the qkv fp32 split-K workspace [16, 2304]-shaped as bf16 does and a zeroed int32 ticket vector, store them in state (never module globals; o has SPLIT_K 1 and needs none), warm each kernel at every T in 2..16 so the timed call never compiles, print '[swapab16-dl] layer=qkv|o T=.. stages=.. n_regs=.. wgmma=.. ON|OFF', and self-check each layer against torch.mm on random bf16 inputs at T = 2, 8, 9, 16 with rel L2 <= 1e-2, then once more after alternating calls on state 0 and state 1 with refilled inputs (ticket reset check). Any failure or OutOfResources at both stage counts turns that layer off (cuBLAS _gemm kept). kernel_fn: a branch on cfg tokens and the state flag only; nothing else changes (attention, boundaries, dense chain P/R, Marlin, unpooler untouched). WHY NOW: LESSONS records that decode's earlier bf16-shape-gated-triton-gemms-small-t won kernel sums at T=8/16 yet lost 2-4.5% on those shapes, 'possibly interference with the side-stream router overlap (unverified)'. small-t-launch-minimal-dense-chain-pdl has since removed the router side stream at T <= 32, so this run is also the direct test of that open lesson: per-layer kernel rows faster AND shapes faster confirms it; rows faster but shapes flat/slower refutes it.

## Expected

- `*load=verify_1 *` faster: T=8: cuBLAS qkv nvjet_sm90_tst_128x8_64x12_4x1 splitK 13.5 us + splitKreduce (~1.7) and o rows (nvjet 64x8 ~7 us) in profile 3fd3fa1f; bf16 m16 swap-AB gave qkv -18%, o -10-19% at this m -> ~3-4 us of 191/202 us, ~2%; read the _swapab_qkv_kernel / _swapab_o_kernel rows against the cuBLAS rows they replace
- `*load=decode_16*` faster: T=16: nvjet 64x16 2x1 (26.2 us over 2 calls) + 64x16 4x1 splitK 12.2 us in profile 3fd3fa1f; same per-kernel saving, ~3-5 us of 205/309 us (sliding weight 46)
- `*load=decode_1 *` flat: T=1 outside the gate (dense-chain-s3 territory)
- `*load=decode_32*` flat: T=32 outside the gate
- `*load=verify_16*` flat: T=128 outside the gate
- `*load=verify_32*` flat: T=256 outside the gate

## Risk

Headline effect is small (~0.3-0.6%), inside drift; judge on the two named kernel rows and the paired per-shape ratios. If rows are faster but verify_1/decode_16 are flat or slower, the savings of these two GEMMs are below drift (this run does not test the old side-stream lesson: the pooler, the GEMM that ran beside the side stream, is out of scope; a pooler-only follow-up is that test). Overlaps the held prefork-qkv-o-swapab-t16-t32 only at T=16; if this lands, retire that file or narrow it to 17..32. Shape-disjoint from dense-chain-s3 (T=1) and sliding-cluster-c4-s3 (attention only), so it can be restacked with either.

## Adversarial review

Reviewed 2026-10-25T111209Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound, rules-clean, shape-gated port of two measured bf16 swap-AB kernels. Its headline effect (~0.4-0.6% geomean) is below what one run can resolve, so read it from the _swapab_qkv_kernel/_swapab_o_kernel rows and the verify_1/decode_16 per-shape ratios. Base it on dense-chain-s3-t1-q-sk2-o-route-unpool (commit e4b88414), not master: the shapes are disjoint and the scored submit then also re-measures that 1.3017x lineage. Drop the claim that this run tests the old side-stream lesson.

### Justification
The mechanism is plausible. On master at T=8 and T=16 the qkv_gate GEMM ([T,8192]x[2304,8192]) and the o GEMM ([T,1024]x[8192,1024]) run as cuBLAS nvjet tiles with split-K plus a splitKreduce node. Profile 3fd3fa1f shows nvjet_sm90_tst_128x8_64x12_4x1 splitK at 13.0 us (7.1%) on sliding verify_1. At decode_16 it shows nvjet 64x16 2x1 at 26.3 us (likely pooler + o, not qkv, so attribute carefully) and a 64x16 4x1 splitK row of ~12 us. Swap-AB puts 64 weight rows on wgmma M and 16 tokens on N, a real tensor-core-path fix at m<=16, and bf16_proj measured it at exactly these weight shapes. Arithmetic: qkv ~14.7 us (nvjet + reduce) -> ~11.5-12 us, o ~7 -> ~6 us, so ~3-4 us off 191/202 us at verify_1 and ~3 us off 205/309 us at decode_16. By TASK.md load weights (46 sliding / 16 full) that is ~1.8% on the verify_1 load and ~1.3% on decode_16, ~0.5% on the 6-load geomean. That is inside the >=1.5% CI, as the hypothesis itself concedes. The expected directions match the gate: T=8 and 16 faster; T=1, 32, 128 and 256 untouched. qkv's 36x4 = 144-CTA grid is just over one wave on 132 SMs, which bf16_proj already absorbed. WEAKEST LINK: the '-18% qkv / -10-19% o' figures were measured against muse_bf16_proj's incumbent at m<=16, which was its own Triton split-K kernels, not cuBLAS nvjet. Against nvjet the saving is inferred, not measured. In decode's earlier bf16-shape-gated run the Triton _qkv_splitk_sem_kernel (14.0 us) was roughly at parity with nvjet + reduce, so the -18% probably carries over to ~-2.5-3 us, but that is an inference. SECOND WEAK LINK (premise, not mechanism): the 'WHY NOW' framing is wrong. In bf16-shape-gated-triton-gemms-small-t only the POOLER ran beside the side-stream router. qkv and o ran on the main stream with nothing concurrent even then, and that run's reviewer named pooler/router contention as the likely loss. This run touches neither the pooler nor the router, so it cannot confirm or refute the side-stream lesson. 'Rows faster but shapes flat' would not justify 'stop porting bf16 small-m GEMMs to this layer'; it would only say these two GEMMs' savings are below drift.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl: it changed the T<=32 launch sequence (main-stream P/R chain with PDL, no router side stream). qkv/o are upstream of the chain and not PDL-linked to it: attn_output feeds the SGLang attn boundary before P. A Triton o kernel that never triggers launch_dependents can only cost the overlap a cuBLAS kernel might give the next node, at most ~1 us. It does not create a fake gain, and it makes the bf16_proj per-kernel numbers more transferable than in the old run.
- full-decode-splitkv-restack-on-pdl (and, if based on s3, full-decode-sk-ticket-restack-b1-b64): it produces o at full decode_16. The o GEMM's input layout and contiguity are unchanged (o is reshaped [T, hq*d] bf16 either way), so there is no interaction beyond L2 state, which the judge colds before each replay. Not a confounder of the qkv/o rows.
- sglang-pipeline-parity-router-hilo-m8 / stack-align-par-sidestream-sliding1: at T>=2 the side stream still carries the expert-dispatch prep (T>32 only) and nothing at T<=32. Nothing runs concurrently with qkv/o at T=8/16, so these do not confound.

### Interference and A/B needs
No companion is warranted. Both outcomes are readable from this one run. A gain shows as the nvjet 128x8 splitK, 64x16 4x1 splitK and splitKreduce rows disappearing at verify_1/decode_16, replaced by _swapab_qkv_kernel and _swapab_o_kernel rows, with per-shape ratios versus the base run's same shapes. A loss is also readable: either the new rows are slower than the nvjet rows (the kernel is wrong for this context) or the rows are faster and the shape is slower. The second case is a launch/PDL-neighbour effect that only an o-only/qkv-only control would split, and that control is worth minting only if that signature appears. Decode has 0 checks left, so this goes out as a scored submit, and a second scored submit spent in advance on a ~0.5% geomean effect is poor value. ab-revert of the dense chain is not meaningful: it reintroduces the router side stream and the 27-29-node graph, which answers a different question. BASE: build on hyp/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool (impl e4b884143211, harvested 1.3017 [1.282,1.321], job 34a9be06918d448b), not master, with depends_on set to it. s3's three pieces are all gated tokens == 1 and this change is gated 2 <= T <= 16, so the gates are disjoint. In s3 the qkv and o call sites at T>1 are still the plain _gemm(...) inside its restructured attention helper, so this is a hand-merge into that helper, not the master kernel_fn line. Because this run is a scored submit, stacking means it also re-measures s3's T=1 gains and the sk-ticket full-decode_16 path (298 vs master 309 us). That lineage's CI barely overlaps BEST (1.282 vs 1.2917 upper), so a second measurement there could actually promote. Read this change's own effect paired against job 34a9be06918d448b's verify_1 and decode_16 rows, not against BEST. Note that s3's sliding decode_16 (209.2 us, err 3.2) sits ~2% above master's 204.7, so treat <=2% on that shape as noise and lean on the kernel rows. The cost is that a failure in s3's CUDA pieces would also sink this submit, but s3 passed 13/13 with the same code, and its pieces fall back per-piece.

### Open questions
- Which GEMMs make up nvjet_sm90_tst_64x16_64x16_2x1 (26.3 us, '2 calls') at decode_16: pooler + o, or qkv + o? This decides whether the decode_16 saving is ~3 us or ~1.5 us. Read the profile's per-call rows before setting the kill threshold.
- Was bf16_proj's '-18% qkv' measured against its Triton split-K incumbent or against cuBLAS? If against Triton, the expected saving against decode's nvjet + splitKreduce is inferred. The run's own rows settle it.
- If based on s3: does s3's attention-helper refactor route full decode_16's o through the same _gemm call as sliding? Port both sites, then confirm in the profile that no nvjet 64x16 o row remains at decode_16 full.
- Delete or narrow the 'stop porting bf16 small-m GEMMs' conclusion in the hypothesis's Risk section. The run does not test the side-stream lesson, because the pooler, which was the GEMM beside the old side stream, is out of scope. A pooler-only follow-up (bf16-m16 swap-AB pooler at T 2..16, now that nothing runs beside it) is the actual test.
- If this lands, narrow the held prefork-qkv-o-swapab-t16-t32 to 17..32 (or retire it), as the hypothesis already notes.

## Implementation notes

Worktree: .worktrees/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork, `muse_decode_layer/submission.py` only, uncommitted. Diff vs the s3 tip file: +447/-3.

**Base.** The worktree was on master's file (54d3a9ce; master dfc610f4 has the same submission.py as merge-base 06cbcc08). Verified the s3 branch file (tip == impl e4b88414) is master + the s3 patch (the +/- lines of `git diff master <s3 branch>` equal the done/neutral .patch's), then copied it in whole, then made this change on top. Gates are disjoint: s3 is tokens == 1 (`dl3["a"] and tokens == 1` in kernel_fn, `use_dl and tokens == 1` for the unpooler), this is 2 <= tokens <= 16.

**Change (all in `_parent_attention` plus a new block before it and one call in `prepare()`).**
- `_parent_attention` is the s3-restructured helper both layer types go through at T > 1. It has exactly one qkv call (`_gemm(data["attn_input"], dense["qkv_gate"])`) and one o call (`_gemm(o, dense["o"])`, shared by the full decode_16 sk/split-KV/FA3 branches and the sliding branch; there is no second o site), both now `if sw_on and sw[layer]: _sw16_* else: _gemm`. `sw_on = state["sw16"] is not None and 2 <= tokens <= 16` with `tokens = cfg["tokens"]`; nothing else is read (no input values, shapes or strides).
- New "swapab16" block: constants, `_swapab_qkv_kernel` and `_swapab_o_kernel` (two verbatim copies of muse_bf16_proj master's `_bf16_swapab_kernel`, generated by script from that file: do_not_specialize M, ticket split-K reduce with self-reset), SWAPAB16 config rows (qkv 2304x8192 SPLIT_K 4 BLOCK_K 128 4 stages; o 8192x1024 SPLIT_K 1 BLOCK_K 256 4 stages), BLOCK_W 64, 4 warps, BLOCK_T = WS_M = 16, fallback to 3 stages on OutOfResources, `_sw16_launch` (the `_swapab16_forward` equivalent).
- `prepare()` calls `_sw16_setup(state, device)` after `_dl3_finish` (so s3's own checks are untouched and run before this flag exists).

**Review points applied.**
1. Workspaces: `state["sw16"]` holds the qkv fp32 `[4, 16, 2304]` workspace and the int32 `[36]` tickets, per state, zero-allocated in prepare(); no module-global tensors, no memset in kernel_fn (the last-arriving CTA resets its ticket). o needs none.
2. Warm-up: every T in 2..16 for each kernel, on live-like layouts: qkv on a contiguous [T, 8192] attn_input; o both as `reshape(T, 1024)` of a [T, 16, 64] tensor (full decode: the sk / split-KV output reshape) and as the real `state["o_norm_gate"]` output given a strided gate view of a [T, 2304] qkv-like tensor (sliding / FA3 path). Prints the compiled-specialization count before and after (M is do_not_specialize, so one specialization is expected per kernel). Not warmed: a 16-byte-misaligned base pointer (no live site produces one; sk / norm-gate outputs and attn_input are fresh allocations).
3. ON lines: per layer, `[swapab16-dl] layer=qkv|o T=2..16 stages=.. n_regs=.. n_spills=.. wgmma=.. mma.sync=.. [errs, graph times] ON`, or `... OFF: <reason> (cuBLAS _gemm kept)`; plus a `PATH CHOSEN` line.
4. Self-check: at T = 2, 8, 9, 16 on both o layouts, against an fp64 reference (a superset of fp32) of the same inputs; bound calibrated per case on cuBLAS's own error: tol = clamp(2 x err_cublas, 4e-3, 1e-2); finiteness required. Then a two-state check: this state's buffers and an independent second set, one graph captured on each at T = 9 and 16, six alternating replays with refilled inputs, each compared to its own reference, tickets must read 0. Each launch is announced with a flushed print before it and synchronized + printed after it.
5. Gate input: cfg["tokens"] only.
6. Both o sites: one call site covers full and sliding in the s3 helper (confirmed by grep: `_gemm(` appears only at the qkv and o lines of `_parent_attention`); the sliding T=1 fused path and full T=1 s3 path are outside the gate.
7. Risk text: the side-stream claim is removed and the Risk section rewritten (above).

**Additions beyond the spec.** A per-layer timing selector (lesson 2026-10-25 03:41Z: every PATH CHOSEN must compare timings): graph medians with a cold-L2 flush, T=8 + T=16 summed, layer stays ON only if swap-AB <= 1.05 x cuBLAS. Prepare timings undercount the profile and are a ranking only, so a spurious OFF is possible (it falls back to cuBLAS = the base).

**Open questions from the review.** The nvjet 64x16 2x1 attribution at decode_16 (pooler + o vs qkv + o) and whether bf16_proj's -18% was against its Triton incumbent need the profile; not answerable at build time. The held prefork-qkv-o-swapab-t16-t32 narrowing (17..32) is a post-landing action, not a build one.

**Not run.** No GPU here: nothing executed. Biggest risk: first Triton compile of two wgmma kernels inside prepare() (compile time, the nested-graph captures in the alternation check) and the timing selector turning a layer OFF on prepare-time noise.

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds only `state.get`, a tokens comparison and two Triton launches with `torch.empty` outputs (capturable, no sync, no `.item()`/`.cpu()`, no input-value branching, no print); all syncs/prints are in prepare().
- [x] Nothing derived from `data` is cached across calls: the workspace, tickets and flags are prepare()-time per state; the gate is cfg["tokens"].
- [x] Weights: unchanged and read in place (BF16 stays BF16); no repack.
- [x] Activations BF16 where SGLang keeps BF16; the kernels read bf16 and accumulate fp32 then round to bf16 like cuBLAS; KV cache and FP8 query paths untouched.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess, fork, or multiprocessing (grep of the file for fork/subprocess/multiprocessing/Popen: only s3's existing comment text).
- [x] All work happens inside the timed call; no output caching or replay: both GEMMs run every call; prepare() only warms, checks and times on synthetic data.
- [x] Per-state buffers: workspace and tickets are in state["sw16"], never module globals; self-check alternates two buffer sets with refilled inputs on captured graphs.
- [x] Marker kept: first line `# okc: problem=muse_decode_layer`.

### Lint

`undef.py`: `muse_decode_layer/submission.py []` (no undefined names). `py_compile`: ok. `okc_lint.py --problem muse_decode_layer`: `OK with warnings` (all warnings are `host-sync` in prepare()-only code outside kernel_fn's call graph: the s3/master ones plus the new block's `torch.cuda.synchronize()` / ticket reads in `_sw16_*` setup helpers (L6147-L6370); none in kernel_fn or in `_parent_attention`). __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus) against the s3 tip file (hyp/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool, e4b88414); diff is the docstring note, one `_sw16_setup` call at the end of prepare(), the swapab16 block, and the two gated call sites in `_parent_attention`.

**Verdict: dispatch** (no fixes).

- Kernels: both `_swapab_qkv_kernel` and `_swapab_o_kernel` are byte-identical (modulo name) to master muse_bf16_proj `_bf16_swapab_kernel`; configs qkv (2304, 8192, SK 4, BK 128, 4 st) grid 36x4, o (8192, 1024, SK 1, BK 256, 4 st) grid 128x1 divide exactly; smem 80 KB / 160 KB per CTA, under the H200 limit, with the 3-stage OutOfResources fallback.
- Gate: `2 <= cfg["tokens"] <= 16` plus the state flags only; disjoint from s3 (tokens == 1). Live T in band: verify_1 (8), decode_16 (16), tests batch 8, 9, 16 and batch 2 x q_len 8; T=1 sliding-fused and full dl3 paths are outside the gate.
- kernel_fn hygiene: kernel_fn path adds `state.get`, a comparison, `torch.empty` and one Triton launch per GEMM; the only try/except in `_sw16_launch` sits under `diag is not None` (prepare-only). No prints, syncs or per-call state. okc_lint: OK with warnings (prepare-only host syncs).
- Graph/compile: M is do_not_specialize; strides (8192 / 1024, weight K, out N) are constant; live inputs (reference attn_input fresh contiguous; sk / split-KV reshape and FusedRMSNormGate output, both warmed) are fresh allocations, so the warmed specialization covers the timed call. Tickets self-reset (last CTA atomic_xchg 0), checked after two-state alternation and zeroed after setup.
- Per-state buffers: ws_qkv / cnt_qkv live in state["sw16"]; no module-global tensors, no pointer-equality gating; even a shallow-shared dict would be safe (sequential same-stream use, self-resetting tickets).
- Self-check tolerance clamp(2 x err_cublas, 4e-3, 1e-2) against fp64 is passable (bf16 output rounding ~1.1-1.6e-3 for both paths), not an always-fail gate. Every failure path in `_sw16_setup` is caught per layer and lands on cuBLAS with an OFF line plus PATH CHOSEN.
- Timing selector: graph medians, cold-L2 flush, same `_gemm` as the live fallback, T=8 + T=16 summed, ON within 1.05x of cuBLAS: sound; a spurious OFF needs a >5% prepare-time loss and only re-measures s3.
- undef.py: none; py_compile ok; __pycache__ removed.

Residual risk: first-time Triton wgmma compile or an OFF from the 1.05x selector leaves the run a re-measure of s3; no crash path found in kernel_fn.

## Result

- job `b0057d2e6cef4d5a` (submit), status done, passed True, tests 13/13
- speedup 1.3056x, 95% CI [1.2860, 1.3252], score 17325.911594027148 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2725637226942819, "ci": [1.2534752668538676, 1.291652178534696], "job": "3fd3fa1f96d045dd"}; base result: muse_decode_layer/hypotheses/done/neutral/w2-marlin-atomic-add-t1-port.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/qkv-o-swapab16-t2-t16-no-fork.result.json`, `muse_decode_layer/hypotheses/done/neutral/qkv-o-swapab16-t2-t16-no-fork.patch`; branch `hyp/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 123.88 | 119.9 | 1.0404 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 139.18 | 130.71 | 1.0793 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 204.68 | 204.72 | 1.005 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 308.69 | 296.65 | 1.0504 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 240.67 | 238.79 | 1.0005 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 473.31 | 440.03 | 1.0864 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 191.42 | 191.58 | 1.0033 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 201.63 | 194.62 | 1.01 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 414.6 | 438.17 | 1.0125 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 565.16 | 576.69 | 1.0114 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 600.78 | 604.04 | 0.9691 | lost |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 880.71 | 883.3 | 0.9767 | flat |

gained on sliding decode_1; full decode_1, decode_16, decode_32; lost on sliding verify_32. Expected hit: 3/6. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: qkv-o-swapab16-t2-t16-no-fork`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

