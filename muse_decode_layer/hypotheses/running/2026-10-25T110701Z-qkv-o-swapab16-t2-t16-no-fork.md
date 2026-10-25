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
status: running
review_at: 2026-10-25T111209Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T111210Z
base_sha: 54d3a9ce8a155b629ac50f80222a1bf0dec23a79
base_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
branch: hyp/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork
worktree: .worktrees/muse_decode_layer/qkv-o-swapab16-t2-t16-no-fork
mode: null
restack_of: null
stale_base: null
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

Headline effect is small (~0.3-0.6%), inside drift; judge on the two named kernel rows and the paired per-shape ratios. If rows are faster but verify_1/decode_16 are flat or slower, record that the T=8/16 loss was NOT the side-stream (the side stream is gone) and stop porting bf16 small-m GEMMs to this layer. Overlaps the held prefork-qkv-o-swapab-t16-t32 only at T=16; if this lands, retire that file or narrow it to 17..32. Shape-disjoint from dense-chain-s3 (T=1) and sliding-cluster-c4-s3 (attention only), so it can be restacked with either.

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

