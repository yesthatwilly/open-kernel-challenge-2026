---
problem: muse_decode_layer
slug: combo-measured-rows-slide-batched-preattn-t8-tail
created: 2026-10-25T161651Z
base: master
area: "combination: sliding attention B=2..32 + pre-attention feeds (T>1) + routing R / unpooler (2<=T<=32 / 2<=T<=8)"
change: >
  COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached
  master, each routed only to the shape class where it won, on master 6773ec5cf5 (dense-chain-s3 +
  dl3 piece A, 1.3080 job 933d3f825a044d36). No new load_inline module (all three sources are
  Triton or already-built dense-chain kernels). SOURCE S1 sliding-fused-batched-decode-port
  (branch hyp/muse_decode_layer/sliding-fused-batched-decode-port, impl 3ddcc4f2a8, job
  b01fd2df41a040b3) in the rebased form of sliding-batched-triton-b16-band (branch
  hyp/muse_decode_layer/sliding-batched-triton-b16-band, impl 1cabea713e, job 02bf417257e14a3d):
  port _sliding_b16_gate and _sliding_forward_b16 (the existing _sliding_fused_kernel launched
  with grid (batch, hkv, 16), S=16 hard-coded, BLOCK_N 64, num_warps 4, num_stages 2) and widen
  the band to two classes: 'sliding q_len 1 page 1 2 <= B <= 16' and '17 <= B <= 32'. Measured:
  sliding decode_16 +3.8% / +5.5% / +5.2% and sliding decode_32 +3.1% / +4.1% / +4.8% against
  their own bases (b01fd2df, 2193fc0d, b94e7973), still +1.7..+4.1% / +2.6..+4.3% today by paired
  SGLang ratio vs master 933d; the incumbent on both rows is unchanged since (FA3 13.6/18.2 +
  combine 4.3/2.9 + prepare_varlen 2.3/2.2 + FusedQKNorm, casts, store_cache, FusedRMSNormGate).
  Route: in _parent_attention after the batch-1 branch, 'elif state flag slide_b16 and class A'
  and 'elif slide_b32 and class B' -> _sliding_forward_b16 with state['slide_cnt']. Check: o
  rel-L2 <= 1e-2 vs master's chain and cache bytes vs master (k within one E4M3 code on < 6% of
  bytes, v exact) at B in {2, 9, 16, 17, 32}, lens {1, 63, 2047, 2048, 2049, 80000}, kv_scale {1,
  0.015625}. Selector per class at B=16 and B=32: the whole attention side (qkv -> o-GEMM input).
  SOURCE S2 stack-route-topk-pdl-unpool-nf1 (branch
  hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1, impl 2fe4d35e13, job 87211e1362b64c4c):
  its two T>1 pieces, which master's dl3 B/C gate to tokens == 1: (S2a) routing R variant 0
  (per-warp top-8, PDL after P) at 2 <= T <= 32 instead of stage-1 R
  (dl_route_norm_align_v1_kernel, 7.3-7.9 us in every T=8/16/32 row of 933d); its own selector
  chose it on the sum of P->R pairs at T=1/8/32 (87.3 vs 93.0 us); (S2b) dl_unpool_kernel NF=1 at
  2 <= T <= 8 (= the queued dl3-unpool-nf1-t2-t8 widening) instead of _gemm(y, unpooler[:1024]) +
  splitKreduce + F.pad. Measured: sliding verify_1 +4.3% vs its base and +2.9% / +3.1%
  (sliding/full verify_1) today vs 933d. Both kernels already live in master's
  muse_decode_dense_chain_s3 module; the change is gates + per-state flags dl3['b_t'] = {8, 16,
  32} and dl3['u8'], extending _dl3_check_r / _dl3_check_u to T in {2, 5, 8, 16, 32}
  (bit-identical ids and align, near-tie rule as master's; unpool rel-L2 <= 2e-3 and exact zero
  columns) and adding selectors _dl3_select_r at T=8/16/32 (P->R pair) and _dl3_select_u at T=8,
  plus the T=8 eager layer check and the T=8 two-state alternation through _dl3_layer_checks.
  SOURCE S3 fused-preattention-paired-pad-restack (branch
  hyp/muse_decode_layer/fused-preattention-paired-pad-restack, impl 963bcf0267, job
  5e9b5464d1444708): port _pre_attention_kernel (PAIRED and unpaired) and
  _paired_norm_gate_kernel. It shipped with NO gate (every T > 1); here route it per class only
  where the attention side still runs FusedQKNorm + k/v/q FP8 casts + store_cache after S1: (c1)
  full decode 1 < B <= 64 (feeds master's sk single-level ticket kernel its FP8 q; master's
  split-KV A/B fallback too), (c2) sliding verify T=8/128/256 (unpaired FA3 feed), (c3) full
  verify B > 1 (PAIRED feed into paired FA3 + _paired_norm_gate_kernel), (c4) full verify_1
  (unpaired). Measured: full decode_16 +4.7%, full decode_32 +3.3%, sliding verify_1 +3.7% against
  its own base. Check: q (bf16 before the cast) rel-L2 <= 1e-3 vs master, cache k bytes by rel-L2
  vs an accurate-rope fp32 reference (LESSONS: Triton sin/cos), v exact, paired halves exactly
  zero, kv_scale {1, 0.015625}, T in {2, 8, 16, 32, 40, 128, 256}. Selector per class at T in {16,
  32} (c1), {8, 128, 256} (c2), {128, 256} (c3), 8 (c4) on the attention side. Every other shape,
  and every cell whose gate fails, is master byte for byte (sliding decode_1, full decode_1 piece
  A, T>32 MoE, all GEMMs). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host
  sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config
  only), no data-dependent Python control flow and no per-call mutable Python state; gates read
  config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as
  given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3;
  every key attended; no child processes. The judge prepares TWO state copies and rotates them:
  every device pointer, counter/ticket array, workspace and TMA map lives in the state, never in a
  module global (this includes the Triton sliding kernel's split counters: master's module-global
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
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "S1: FA3 chain (~28 us of nodes in 933d profile 2) -> one _sliding_fused_kernel launch (attention: 13.0 us); measured -8..-11 us before; plus S2a R 7.6 -> ~6 us. 195.6 -> ~185 us"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "S1 at B=32 (measured +3.1..+4.8%); S2a; 239.2 -> ~229 us on master's worst-ratio row (1.080x)"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: S2b unpool T=8 (nvjet 64x8 splitK 5.0-5.5 + splitKreduce + F.pad -> ~6.5 us kernel) + S2a R + S3 c2/c4 (5 launches -> 1); measured +2.9/+3.1% today; ~-5..-8 us
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: S3 c1 feeds sk directly (measured +4.7% on its base) + S2a; ~-3..-6 us on 296.7
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: S3 c1 + S2a; ~-3..-5 us on 430.7
  - shapes: "*load=verify_16*"
    direction: flat
    why: S3 c2/c3 only (measured flat there); control for drift
  - shapes: "*load=verify_32*"
    direction: flat
    why: S3 c2/c3 only; control
  - shapes: "*load=decode_1 *"
    direction: flat
    why: untouched (dl3 / _sliding_forward); drift control
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T162256Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T162256Z
base_sha: 0c4719336efd5dcbef40bf2e1bbbccad0ca34463
base_blob: 4bb57fd292a7f0fd5a80bd88dc773008b490bf8c
branch: hyp/muse_decode_layer/combo-measured-rows-slide-batched-preattn-t8-tail
worktree: .worktrees/muse_decode_layer/combo-measured-rows-slide-batched-preattn-t8-tail
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T191043Z
impl_commit: b257ef2b43b4e7fc3b6a738c33f35c155392c3ed
submission_sha256: 56bed064ba73e4549099978a1dd05461d071cbf94032c3984f1da8cc58a2917d
norm_sha256: b2394dd23614aa30635bb318b5cb32306e73ccbfce09cfac4da9071796e51f26
submission_blob: 3fc72e97304e132db432fcc87872be997f842929
lint_warnings: 71
attempt_file: 2026-10-25T191044Z-submit-combo-measured-rows-slide-batched-preattn-t8-tail.md
prepared_at: 2026-10-25T191044Z
job: a903b9d004d2452d
dispatched_at: 2026-10-25T191052Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.3142361220457452
speedup_ci: [1.294522580215059, 1.3339496638764312]
harvested_at: 2026-10-25T192026Z
best_at_harvest: 1.3079569800910293
promotion: merged
hand_landed: true
promotion_commit: ec89382beb4990b340cef17f37a963295ee61c0f
---

# combo-measured-rows-slide-batched-preattn-t8-tail

## Change

COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached master, each routed only to the shape class where it won, on master 6773ec5cf5 (dense-chain-s3 + dl3 piece A, 1.3080 job 933d3f825a044d36). No new load_inline module (all three sources are Triton or already-built dense-chain kernels). SOURCE S1 sliding-fused-batched-decode-port (branch hyp/muse_decode_layer/sliding-fused-batched-decode-port, impl 3ddcc4f2a8, job b01fd2df41a040b3) in the rebased form of sliding-batched-triton-b16-band (branch hyp/muse_decode_layer/sliding-batched-triton-b16-band, impl 1cabea713e, job 02bf417257e14a3d): port _sliding_b16_gate and _sliding_forward_b16 (the existing _sliding_fused_kernel launched with grid (batch, hkv, 16), S=16 hard-coded, BLOCK_N 64, num_warps 4, num_stages 2) and widen the band to two classes: 'sliding q_len 1 page 1 2 <= B <= 16' and '17 <= B <= 32'. Measured: sliding decode_16 +3.8% / +5.5% / +5.2% and sliding decode_32 +3.1% / +4.1% / +4.8% against their own bases (b01fd2df, 2193fc0d, b94e7973), still +1.7..+4.1% / +2.6..+4.3% today by paired SGLang ratio vs master 933d; the incumbent on both rows is unchanged since (FA3 13.6/18.2 + combine 4.3/2.9 + prepare_varlen 2.3/2.2 + FusedQKNorm, casts, store_cache, FusedRMSNormGate). Route: in _parent_attention after the batch-1 branch, 'elif state flag slide_b16 and class A' and 'elif slide_b32 and class B' -> _sliding_forward_b16 with state['slide_cnt']. Check: o rel-L2 <= 1e-2 vs master's chain and cache bytes vs master (k within one E4M3 code on < 6% of bytes, v exact) at B in {2, 9, 16, 17, 32}, lens {1, 63, 2047, 2048, 2049, 80000}, kv_scale {1, 0.015625}. Selector per class at B=16 and B=32: the whole attention side (qkv -> o-GEMM input). SOURCE S2 stack-route-topk-pdl-unpool-nf1 (branch hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1, impl 2fe4d35e13, job 87211e1362b64c4c): its two T>1 pieces, which master's dl3 B/C gate to tokens == 1: (S2a) routing R variant 0 (per-warp top-8, PDL after P) at 2 <= T <= 32 instead of stage-1 R (dl_route_norm_align_v1_kernel, 7.3-7.9 us in every T=8/16/32 row of 933d); its own selector chose it on the sum of P->R pairs at T=1/8/32 (87.3 vs 93.0 us); (S2b) dl_unpool_kernel NF=1 at 2 <= T <= 8 (= the queued dl3-unpool-nf1-t2-t8 widening) instead of _gemm(y, unpooler[:1024]) + splitKreduce + F.pad. Measured: sliding verify_1 +4.3% vs its base and +2.9% / +3.1% (sliding/full verify_1) today vs 933d. Both kernels already live in master's muse_decode_dense_chain_s3 module; the change is gates + per-state flags dl3['b_t'] = {8, 16, 32} and dl3['u8'], extending _dl3_check_r / _dl3_check_u to T in {2, 5, 8, 16, 32} (bit-identical ids and align, near-tie rule as master's; unpool rel-L2 <= 2e-3 and exact zero columns) and adding selectors _dl3_select_r at T=8/16/32 (P->R pair) and _dl3_select_u at T=8, plus the T=8 eager layer check and the T=8 two-state alternation through _dl3_layer_checks. SOURCE S3 fused-preattention-paired-pad-restack (branch hyp/muse_decode_layer/fused-preattention-paired-pad-restack, impl 963bcf0267, job 5e9b5464d1444708): port _pre_attention_kernel (PAIRED and unpaired) and _paired_norm_gate_kernel. It shipped with NO gate (every T > 1); here route it per class only where the attention side still runs FusedQKNorm + k/v/q FP8 casts + store_cache after S1: (c1) full decode 1 < B <= 64 (feeds master's sk single-level ticket kernel its FP8 q; master's split-KV A/B fallback too), (c2) sliding verify T=8/128/256 (unpaired FA3 feed), (c3) full verify B > 1 (PAIRED feed into paired FA3 + _paired_norm_gate_kernel), (c4) full verify_1 (unpaired). Measured: full decode_16 +4.7%, full decode_32 +3.3%, sliding verify_1 +3.7% against its own base. Check: q (bf16 before the cast) rel-L2 <= 1e-3 vs master, cache k bytes by rel-L2 vs an accurate-rope fp32 reference (LESSONS: Triton sin/cos), v exact, paired halves exactly zero, kv_scale {1, 0.015625}, T in {2, 8, 16, 32, 40, 128, 256}. Selector per class at T in {16, 32} (c1), {8, 128, 256} (c2), {128, 256} (c3), 8 (c4) on the attention side. Every other shape, and every cell whose gate fails, is master byte for byte (sliding decode_1, full decode_1 piece A, T>32 MoE, all GEMMs). HARD RULES (every path): CUDA-graph capturable; kernel_fn has no host sync, no timing, no prints, no try/except, no allocation sized by input values (shape/config only), no data-dependent Python control flow and no per-call mutable Python state; gates read config/shape and this state's own prepare() flags only; nothing cached from inputs; weights as given (NVFP4 stays NVFP4, dequant in registers only; BF16 stays BF16); KV FP8, q at most E4M3; every key attended; no child processes. The judge prepares TWO state copies and rotates them: every device pointer, counter/ticket array, workspace and TMA map lives in the state, never in a module global (this includes the Triton sliding kernel's split counters: master's module-global _COUNTERS dict must NOT be used by any new band; allocate state['slide_cnt'] [64*2] int32 per state and slice it). Every new CUDA build is sm_90a only, -O3 -std=c++20 -lineinfo, a renamed load_inline module built at import like master's, with an in-process canary (synchronize + 'launching'/'ok' prints, flush=True) after each first launch; on any miss a 'PATH CHOSEN ... (fallback: <reason>)' line keeps master for that cell. GATE PER SOURCE PER CLASS: each source path has its own prepare-time gate = (a) correctness check (bound stated per source below) on two alternating states with refilled inputs under captured-graph replay, then (b) a cold-L2 CUDA-graph selector (256 MB flush before each replay, >= 41 interleaved reps, empty-graph baseline subtracted) that enables the cell only when new/master <= 0.98 (i.e. master/new >= 1.02) and asserts the enabled flag equals the printed winner. Print exactly one 'PATH CHOSEN: <source> <class>: <winner> [new X us vs master Y us]' line per source per class. Every added prepare() phase runs AFTER master's own setup (after _dl3_finish), each with its own start deadline measured from prepare()'s t_prepare0 (skip the phase with a fallback line if the deadline has passed) and a print of its own wall seconds plus the cumulative prepare wall time; master's own internal deadlines and its _sk_prepare budget do not move. Print compile seconds per load_inline module. First measurement is an okc check.

## Expected

- `*layer=sliding load=decode_16*` faster: S1: FA3 chain (~28 us of nodes in 933d profile 2) -> one _sliding_fused_kernel launch (attention: 13.0 us); measured -8..-11 us before; plus S2a R 7.6 -> ~6 us. 195.6 -> ~185 us
- `*layer=sliding load=decode_32*` faster: S1 at B=32 (measured +3.1..+4.8%); S2a; 239.2 -> ~229 us on master's worst-ratio row (1.080x)
- `*load=verify_1 *` faster: S2b unpool T=8 (nvjet 64x8 splitK 5.0-5.5 + splitKreduce + F.pad -> ~6.5 us kernel) + S2a R + S3 c2/c4 (5 launches -> 1); measured +2.9/+3.1% today; ~-5..-8 us
- `*layer=full load=decode_16*` faster: S3 c1 feeds sk directly (measured +4.7% on its base) + S2a; ~-3..-6 us on 296.7
- `*layer=full load=decode_32*` faster: S3 c1 + S2a; ~-3..-5 us on 430.7
- `*load=verify_16*` flat: S3 c2/c3 only (measured flat there); control for drift
- `*load=verify_32*` flat: S3 c2/c3 only; control
- `*load=decode_1 *` flat: untouched (dl3 / _sliding_forward); drift control

## Risk

Winner map behind this stack (paired SGLang ratio vs master 933d, which re-bases each source to today): Sd16 S1 +1.7..+4.1; Sd32 S1 +2.6..+4.3; Sv1 S2 +2.9, S3 +2.3; Fv1 S2 +3.1, S3 +3.1; Fd16/Fd32 S3 +4.7/+3.3 on its own base (today's paired ratio is from a pre-sk base, so only the own-base delta transfers); every other row master. Expected headline about +1.5..+2% (decode_16 ~3.5%, decode_32 ~2.7%, verify_1 ~3.5% of their loads). Prepare phases (all after _dl3_finish; master's own prepare is 4.5 s in 933d): S2 checks + selectors + T=8 layer checks ~4 s (start deadline 120 s); S1 Triton re-specialisation, checks, 2 selectors ~10 s (deadline 150 s); S3 Triton JIT for paired/unpaired, checks, 8 selector cells ~25 s (deadline 180 s); total ~45 s, no import-time compile added (still 3 modules, ~120 s at import).

## Conjecture

The three sources were each measured on a base where the other two were absent, and S3's full-decode gain was measured feeding FA3 / split-KV, not today's sk kernel. The interaction tested is graph-node removal compounding in the T<=32 dense-chain layer: each source removes 2-5 graph nodes from the same serial stream that master's PDL chain (P -> R -> experts -> unpool) runs on, and at decode_16/32 the layer is node-bound (FINDINGS: ~27-29 nodes; the Triton kernel saved ~9-11 us of the decode_16 row, more than the 13.6 us FA3 kernel minus its own 13.0 us). If the whole-layer row delta exceeds the sum of the per-cell selector deltas (each selector times only its own sub-graph), the removed node boundaries were worth more than their kernels, i.e. launch gaps between neighbours, and the next move is fusing across sources; if it falls short, the selectors overstate (L2 warm from the neighbour in isolation). Confirm/refute: compare each 'PATH CHOSEN' line's (master - new) us with the per-shape us delta vs 933d on the same rows (Sd16, Sd32, Sv1, Fd16), and count nodes in the profile ('... N more' lines) before/after. Second check: S2a's R selector re-decides at T=16/32 now that P is the s3 module's and dl3 piece B owns T=1; a MASTER verdict there refutes the transfer of its T=1/8/32 sum.

## Adversarial review

Reviewed 2026-10-25T162256Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with sharp conditions. Every cell has its own correctness gate and a 0.98 selector, so the worst likely outcome is flat rows, not a loss. The weakest link is S1 at B=2..16: its only direct kernel measurement lost. The Conjecture is only partly testable: its S2a re-decision half is readable, but its node-compounding half sits inside per-row noise.

### Justification
Arithmetic check: at master 933d's rows, -10 us on Sd16, -10 on Sd32, -4..-5 on Fd16/Fd32 and -6.5 on both verify_1 rows give load gains of about 3.9% (decode_16), 2.9% (decode_32) and 3.5% (verify_1), with the other three loads flat. That is a geomean of about +1.7%, which matches the stated +1.5..2% and sits right at the ±1.5% CI. So the run has to be read row by row and from the PATH CHOSEN lines, not from the headline. The expected shape directions are consistent with TASK weights (sliding rows weight 46) and floors. Sd32 is master's worst row (1.080x, 239.2 vs SoL 84.5), so it has the most to gain.

S1 at 17<=B<=32 (class B) has the best evidence. b94e (Sd32 246.99 -> 221.38 on its own base) and b01fd both ran S=16 at B=32. The incumbent there is still the FA3 chain: FA3 18.2 + combine 2.9 + prepare_varlen 2.2 + FusedQKNorm/casts/store_cache/FusedRMSNormGate nodes. Removing about 6 serial nodes on a node-bound T=32 layer is plausible.

WEAKEST LINK: S1 at 2<=B<=16 (class A). The only run of exactly this code form (sliding-batched-triton-b16-band, job 02bf417257e14a3d, on master) measured _sliding_fused_kernel at 14.6 us vs FA3 13.7 us. Its Sd16 row went 209.82 -> 215.34 (+2.6%), and REWRITE_READ_2026-10-25 calls the line DEAD. The '+1.7..+4.1% today by paired SGLang ratio vs 933d' is a cross-run paired-ratio comparison. The same 02bf run reads +1.7% vs 933d while it was -2.6% vs its own base, so that number is baseline drift and not evidence. The Sd16 gains in b94e and b01fd came on pre-sk, pre-dl3 bases whose Sd16 row was 209-217 us; today it is 195.6. Expect the class-A selector to print MASTER, and Sd16 to come in flat rather than at the ~185 us predicted. That is safe: the gate turns it off.

S2 (route R variant 0 at T=2..32, unpool NF=1 at T=2..8): both kernels already live in master's s3 module and passed the judge at T<=32 in 87211e. The R selector was decided on an older P, so its re-decision at T=16/32 is a fair test.

S3 c1 (Fd16/Fd32): the +4.7%/+3.3% was measured feeding paired FA3 / split-KV, not today's sk kernel, and REWRITE_READ found the batched rows of that run within err. Today the c1 saving is at most the q FP8 cast plus the FusedQKNorm/casts/store_cache nodes (~2-4 us on 297/431 us rows, 16-weight). 'Faster' there is weakly supported, so treat flat as the expected outcome.

CONJECTURE: half testable. (1) Comparing the sum of the selector deltas (master - new, from prepare) against the per-row delta vs 933d cannot separate 'exceeds' from 'falls short'. Batched-row errs are 2.4-4.1 us and verify_1 2.9-3.5 us, and cross-run SGLang drift adds 2-7% on untouched rows (verify_32 1.02-1.07 in REWRITE_READ). The expected per-row deltas (5-10 us) are only about 1.5-3 z, so only a gap larger than about 5 us per row would read. (2) The node count before and after from the profile, and S2a's R selector verdict at T=16/32 vs its old T=1/8/32 sum, are directly readable from this run's log and profile. That half confirms or refutes.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1 (landed, 8fa8fbf): it put moe-align-parallel-large-t-port on master. The failed run b94e797333fb4a27 was exactly align-par + the Triton batched sliding path (sliding-fused-batched-decode-port), and it failed at full verify_1 in timing with tests 13/13 and no root cause. This combo re-creates that pairing (align-par on master + S1). It also changes full verify_1 itself through S2a, S2b and S3 c4, so a repeat failure on that row could not be pinned on any one source.
- small-t-launch-minimal-dense-chain-pdl + dense-chain-s3-t1-q-sk2-o-route-unpool: S2a's per-warp R was chosen on the sum of P->R pairs measured against an older P and PDL pairing. The P kernel and its PDL edge are now the s3 module's, and dl3 piece B owns T=1, so the old T=1/8/32 sum does not transfer. The hypothesis already re-decides at T=8/16/32, which is correct. Read the T=16/32 verdict as the answer.
- full-decode-splitkv-restack-on-pdl / sk ticket kernels (in master via dense-chain-s3's base): S3 c1's +4.7%/+3.3% was measured with paired FA3/split-KV as the consumer. With sk as the consumer, only the cast and pre-attention nodes are removed. The gain size is confounded by the consumer swap; the selector decides.
- fa3-hdim128-head-pairing-batched (landed): S3 c3 replaces master's paired q zero-fill + two slice copies and the unpad/FusedRMSNormGate with the PAIRED pre-attention kernel + _paired_norm_gate_kernel. Any verify_16/verify_32 move is S3 c3 interacting with the paired layout. Those rows are expected flat and serve as the drift control, so a move there points at c3.
- dl3-piece-a-gate-stride-check-fix (landed): S2's new phases must run after _dl3_finish, and the extended _dl3_layer_checks must not change piece A's T=1 decision. Sliding and full decode_1 should be byte-identical to 933d; if either moves beyond ~1 us err, the S2 phase changed piece A's or piece B/C's T=1 state.

### Interference and A/B needs
No companion. Each source x class cell has its own printed PATH CHOSEN line with new-vs-master us, and the profile names the kernels: _sliding_fused_kernel, dl_route variant, dl_unpool_kernel, _pre_attention_kernel. The rows split almost cleanly by source:
- Sd16/Sd32 = S1 + S2a
- Fd16/Fd32 = S3c1 + S2a
- verify_1 = S2a + S2b + S3c2/c4
- verify_16/32 = S3c2/c3 only
- decode_1 = untouched control
A gain or a loss on any row is attributable from the log plus the profile without another run. An ab-revert of align-par, to test the b94e pairing, would cost a scored run and still not explain a root cause that the earlier run never logged. The better spend is the extra prepare-time full-layer T=8 check below. A per-source control would only re-measure what the selectors already time.

### Open questions
- b94e797333fb4a27 exposure: the Triton batched path itself never runs on full verify_1, because S1's gate is window != 0 and q_len == 1. Its only cross-shape footprint in b94e was the module-global _COUNTERS, the module-global _SLIDING_FUSED_ON with try/except, and per-call graph-pool workspaces. This combo removes the first two (per-state counters, no try/except), and that is the guard against cross-state ticket corruption. The root cause stays unknown, and this combo changes full verify_1 directly through S2a (R variant 0 at T=8), S2b (unpool at T=8) and S3 c4, so it IS exposed on that exact row. Required guard: the T=8 eager layer check in _dl3_layer_checks must run a FULL-layer config (window 0, B=1, q_len 8, ctx 80000, plus one ctx near 1048576) as well as a sliding one. It should run end to end under two-state alternating graph replay with refilled inputs and apply the judge's routing rules (picked within 0.05 of the 8th best, clear-margin top-8 present, route and distinct-expert counts), not only bit-identity to stage-1 R. Print the max rel error per output. On any miss, turn off S2a/S2b/S3c4 at T=8 for that state.
- If class A (B=2..16) prints MASTER, Sd16 should be flat and its expected 'faster' is refuted by the selector, not by the run. Is the builder keeping class A only because the selector is cheap?
- Does the judge deep-copy prepare()'s state or call prepare() twice? Either way, per-state slide_cnt is correct. Under deepcopy, though, any tensor cached in a module global (the current _COUNTERS and _INV_FREQ) is shared, which is the reason to move both.
- Do the S2 checks cover bsm=16 at T=32 (marlin-block-m-cap-16) for R variant 0's align output? The hypothesis lists T=32 in _dl3_check_r; confirm sorted_ids, block_experts and n_padded are compared there, not only the ids.
- Total wall time: import ~120 s plus master prepare 4.5 s plus ~45 s of new phases, then 13 tests and 12 timed shapes, all inside 15 minutes. Two prepare() calls double the 45 s. Confirm the deadlines (120/150/180 s from t_prepare0) apply per prepare call.


## Implementation notes

Built 2026-10-25 on base 0c4719336e (master's muse_decode_layer/submission.py is byte-identical at c3f81624, so no merge was needed). Only `muse_decode_layer/submission.py` changed: +1390 / -166 lines. The diff adds no new load_inline module.

**Layout for the sibling combos C2 and C3.** Each source has one named prepare phase and one per-state flag:

| Source | Phase | Flag |
|---|---|---|
| S1 | `_cb_prepare_s1` | `state["s1"][A or B]` |
| S2a | `_cb_prepare_s2a` | `state["dl3"]["b_t"][8, 16 or 32]` |
| S2b | `_cb_prepare_s2b` | `state["dl3"]["u8"]` |
| S3 | `_cb_prepare_s3` | `state["s3"][cell]`, with cells c1-b16, c1-b32, c2-t8, c2-t128, c2-t256, c3-t128, c3-t256, c4-t8 |

After those phases, `_cb_prepare_t8` runs the T=8 layer check. `_combo_prepare` calls all of them after `_dl3_finish`, and each phase goes through `_cb_phase`. `_cb_phase` checks the start deadline measured from t_prepare0 (S2a and S2b 120 s, S1 150 s, S3 180 s, T8 210 s; no new cell starts after 255 s). It catches exceptions, prints the phase's wall seconds and the cumulative prepare time, and applies the flags. `_cb_report` then prints one `PATH CHOSEN: <source> <class>: NEW|MASTER ... [new X us vs master Y us]` line for each of the 14 source x class cells. It checks that each flag equals the printed winner and forces master on a mismatch. Master's T>1 attention side is now `_attn_core`: `_master_front` + `_attn_route` + FA3, sk or split-KV. With every flag off it issues master's launches in master's order. `_parent_attention` is `_gemm(_attn_core(...), o)`.

**kernel_fn** reads only config and this state's flags. `_attn_core` routes S1 (`_sliding_forward_band`, S=16, BLOCK_N 64, 4 warps, 2 stages) and S3 (`_s3_pre_attention`, plus `_s3_paired_norm_gate` in c3). kernel_fn itself passes `r_t` to `_dl_forward` for S2a and takes `dl3["u8"]` into the unpool branch for S2b. kernel_fn has no prints, no try/except and no per-call mutation.

**Selectors (all cells).** `_cb_time` captures each path as a graph, flushes 256 MB before every replay, runs 41 interleaved rounds (order reversed every round, after 3 warm rounds) and subtracts the median of a one-kernel graph. A cell is enabled only when new/master <= 0.98. The plain numbers from the first prepare() are reused by the second one (`_CB_SELECT`). Each state still runs its own correctness gates.

**Correctness gates.** Every source x class cell gets two-state alternating graph replay with refilled inputs and poisoned outputs. `_cb_second_state` gives the second state its own slide_cnt, P/R, Q/O/unpool and sk buffers.
- S1 and S3 (`_cb_attn_alt`) compare the attention side (qkv to o-GEMM input) against master run eagerly on a clone, and against an fp64 reference (`_cb_attn_ref`).
- S2a (`_dl3_check_r` at T in {2, 5, 8, 16, 32} plus `_cb_s2_alt` "R") requires all seven R outputs to be bit-identical to stage 1's R. That includes sorted_ids, block_experts and n_padded at T=32 (bsm 16), which answers the review's open question.
- S2b (`_dl3_check_u` at T in {2, 5, 8} plus `_cb_s2_alt` "U") requires rel-L2 <= 2e-3 and exact zero columns.
- S3 also runs a kernel-level `_s3_front_check` on q, k and v, described under review item (4).

### Review fixes applied
1. **S1 is ported only in the sliding-batched-triton-b16-band form (1cabea713e).** `_sliding_b16_gate` became `_s1_class` and `_sliding_forward_b16` became `_sliding_forward_band`, with the band widened to classes A (2..16) and B (17..32). Nothing from sliding-fused-batched-decode-port was taken: no try/except and no `_SLIDING_FUSED_ON`.
2. **Counters and `_INV_FREQ` moved into the state.** `state["slide_cnt"]` is `torch.zeros(128, int32)` allocated in prepare() through `_slide_state`. The band uses [0:B*Hkv] and master's batch-1 `_sliding_forward` now uses its own slice [64:66]. `state["slide_inv_freq"]` replaces `_INV_FREQ`, and `_COUNTERS`, `_counters` and `_inv_freq` are deleted. `_dl3_ministate` also gives the second state its own slide_cnt.
3. **Counter assertion.** `_cb_clean` runs after every check, alternation, selector and phase. If any counter (slide_cnt, cnt_p/r, cnt_q/o/u, both sk ticket levels) is not 0, it zeroes them and sets `recs["_dirty"]`, which keeps master for every combo cell of that state.
4. **S3 RoPE check by rel-L2.** `_s3_front_check` covers position bases 80000, 299995 and 1048500 on the sliding cells, and 1048500 on c4. It compares q and the new k rows by rel-L2 against an fp64 accurate-RoPE reference (`_cb_rope64`) and requires new <= master + 1e-3 and at most one E4M3 code gap from master. It also requires v exact, the paired zero half exactly 0 and every other row untouched.
5. **FULL-layer T=8 check.** `_dl3_layer_checks(..., t8=True)` calls `_cb_t8_layer_check`, which runs kernel_fn at B=1, q_len 8 on three cases: full ctx 80000, full ctx 1048570, and sliding ctx 80000 (the sliding companion the review asked for). Each case runs two states with one graph each and 4 alternating replays with refilled inputs and poisoned outputs.
   - The judge's routing rules are applied against master's fp64 router scores (`_cb_route_rules`): pick within 0.05 of the 8th, clear-margin (> 0.03) local top-8 present, routes >= 92% - 4, distinct >= 85% - 2.
   - It prints the max error per output: hidden, stream, update, k_cache, k_gap and token ids.
   - On a miss, S2a class 8, S2b, S3 c4-t8 and S3 c2-t8 go off for that state. If the check cannot run (deadline or exception), those cells also go off.
6. **Every source x class has its own gate, selector and line.** That is 14 cells, each with its own correctness gate, its own selector at <= 0.98 and one PATH CHOSEN line.

### Answers to the open questions
- **Is class A kept only because the selector is cheap?** Yes. It is gated, and the expected verdict is MASTER, which keeps Sd16 flat.
- **Deepcopy vs a second prepare() call:** this works either way, because every buffer is per state and no module global holds a tensor.
- **Deadlines** are per prepare() call, measured from that call's own t_prepare0.

### Deviations
- **S1/S3 attention-side bound.** The bound is "o rel-L2 <= 1e-2 vs master OR new vs fp64 <= 1.25 x master vs fp64 + 1e-3". The spec says only 1e-2 vs master. Master's FP8-q error alone can exceed 1e-2 on peaked synthetic softmax (LESSONS: calibrate bounds against fp32 and master's own error). Cache bounds follow the spec: k within 1 code on < 6% of bytes, v exact, other rows identical.
- **S3 q check.** The spec asks for q (bf16 before the cast) rel-L2 <= 1e-3 vs master, but the S3 kernel never produces a bf16 q. The check instead compares the FP8 q against the fp64 reference (new <= master + 1e-3) and requires at most one code gap from master.
- **S3 classes as cells.** S3 has 8 cells, one per selector point the spec lists: c1 {16, 32}, c2 {8, 128, 256}, c3 {128, 256}, c4 8. Batch splits at 23/24, as the sk batch classes do. Verify cells are limited to 2 <= q_len <= 8, B <= 64, and include a (B=5, L=3) check. c1 runs only when this state feeds sk or split-KV (unpaired q); otherwise that cell stays on master.
- **Synthetic keys** have std 0.3 (soft) and are clamped to stay inside E4M3.
- **T=8 layer tolerances.** With identical ids: hidden <= 2e-2, stream <= 1e-2, update <= 5e-2. If ids differ but the routing rules pass: 5e-2 / 2.5e-2 / 1e-1. Cache: k <= 1 code and rel <= 2e-2, v exact.

### Biggest risk
Prepare wall time, roughly 60-90 s per call and doubled across two calls:
- Graph captures at ctx 80000, B=32, plus data builds of about 650 MB.
- Full-layer verify at ctx 1048570.
- About 50 S3 front checks and fp64 references.

The per-phase deadlines bound it. First measurement must be an okc check.

## Rules check
- [x] CUDA-graph capturable. kernel_fn, `_attn_core`, `_sliding_forward_band`, `_s3_pre_attention` and `_s3_paired_norm_gate` have no .item(), .cpu(), synchronize, prints or try/except (grep of the diff). Allocations are sized by config only. Branches use config plus this state's prepare() flags only.
- [x] Nothing derived from `data` is cached. The state holds only prepare()-built buffers and flags. `_CB_SELECT` holds plain floats and bools from prepare() timing on synthetic data.
- [x] Weights are untouched (no new repack). NVFP4 and BF16 are as in master.
- [x] Activations: BF16 as master. KV FP8 (S1 and S3 store E4M3). q is E4M3 (S3) or bf16 (S1), never coarser than E4M3.
- [x] Every key is attended. S1 covers the full 2048-token window with 16 splits; S3 changes only the front half before FA3, sk or split-KV.
- [x] No timing patches, harness access, module edits, reference import, processes or network. There is no subprocess, multiprocessing or os.fork anywhere.
- [x] All work is inside the timed call, with no output caching. prepare() checks and selectors use synthetic data only.
- [x] Two-state rotation is safe: there is no module global with a device pointer. `_COUNTERS` and `_INV_FREQ` were removed and slide_cnt is per state.
- [x] n/a: there are no new load_inline, sm_90a or wgmma builds (no new CUDA module); Triton launches are announced and synced in prepare() (`_cb_launch`).

```
python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer
-> OK with warnings (71 host-sync warnings, every one "outside kernel_fn's call graph; fine in prepare()"; 0 errors)
python3 -I -m py_compile -> ok; symtable scan for undefined globals -> none; __pycache__ removed
```

## Pre-flight audit

Audited 2026-10-25 (opus, read-only plus numpy emulation; no GPU). **Verdict: dispatch-with-fixes.**

Verified as claimed:
- With every flag off, `_attn_core` + `_parent_attention` issue master's launches in master's order with master's arguments: qkv GEMM, then the B=1 sliding kernel, or FusedQKNorm, the k/v divide, the FP8 casts and store_cache, then sk, split-KV, or the paired zero-pad copies + FA3 + unpad + FusedRMSNormGate, then the o GEMM. `_attn_route`, `_s1_class` and `_s3_cell` are pure Python on config and flags. kernel_fn's R and unpool conditions reduce to master's when `b_t` and `u8` are False.
- No module global holds a per-state tensor. `_COUNTERS`, `_INV_FREQ`, `_counters` and `_inv_freq` are gone. `slide_cnt` is per state, with the band on [0:B*Hkv <= 64] and B=1 on [64:66]. `_dl3_ministate` and `_cb_second_state` give the second state its own counters. `_CB_SELECT` and `_SMS` hold only plain numbers.
- Only one prepare() call per process has been seen in master logs (cumulative equals this call, 4.5 s). Deadlines are per call, from t_prepare0.

Fixes (worktree `muse_decode_layer/submission.py`):
1. **S1 cache gate could never pass (category 1).** `_cb_cmp_attn` required new k rows within one E4M3 code of master on < 6% of bytes. S1 is master's Triton `_sliding_fused_kernel`, which computes RoPE with tl.cos/tl.sin. If these lower to the approximate MUFU instructions (LESSONS line 140), numpy emulation gives code gaps of 6-7 at position ~2048 with kv_scale 1/64, and 8 (kv 1) or 90 (kv 1/64) at position 80000, because near-zero codes are dense. That would turn S1 off in both classes on every run. The k gate is now rel-L2 of the dequantized new k rows vs master's at <= 4e-2, which is half the judge's 8% cache bound. Emulated: mean 1.6%, max 3.6% on the smallest case. v exact and untouched rows identical are kept. Code gap and fraction are printed as info only.
2. **The o bound could hide one wrong row.** The fp64 alternative (new <= 1.25x master + 1e-3, aggregate) could pass a row about 20% off in a B=32 batch. It now also requires every token row to satisfy new-vs-fp64 <= 2x master's row error + 5e-3. The first disjunct (<= 1e-2 vs master) is unchanged: a fully wrong row in B=32 gives an aggregate error >= 0.18. S3's q check (gap <= 1 code everywhere, zero half exact, no worse than master vs fp64 + 1e-3) already rejects wrong RoPE, heads or rounding, so it is left as is.
3. **The T=8 layer check tripped on noise.** The k rel-L2 bound was 2e-2. fp32->E4M3 (S3) vs bf16->E4M3 (master) double rounding gives about 1.6% mean, with P(>2e-2) of about 7% per replay at 16 rows (emulated). Over up to 12 replays it would turn the verify_1 cells off about half the time. The bound is now 4e-2 (emulated max 2.6%). `k_gap <= 1` is kept, since both paths use the cos_sin table.
4. **Deadlines tightened so a phase skips rather than overruns.** The T8 start deadline goes from 210 to 200 s and the cell deadline from 255 to 240 s. A new 260 s case deadline (`_cb_case_late`, using the plain float `_CB_T0`) is checked before each `_cb_attn_alt` case and each T=8 case. A late check returns a failure, which keeps master for that cell. The worst case is about 270 s, against the 300 s limit.

lint: OK with warnings (prepare-only host syncs). py_compile: ok. __pycache__ removed.

Residual risk: none of this has run on a GPU. Prepare wall time (estimated 60-90 s) and the S2a/S3 kernels' first T>1 launches inside this stack are unmeasured. A slow import plus prepare could squeeze the 15-minute run budget.

## Result

- job `a903b9d004d2452d` (submit), status done, passed True, tests 13/13
- speedup 1.3142x, 95% CI [1.2945, 1.3339], score 17211.74728083805 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.3079569800910293, "ci": [1.288337625389664, 1.3275763347923946], "job": "933d3f825a044d36"}; base result: muse_decode_layer/hypotheses/done/neutral/dl3-piece-a-gate-stride-check-fix.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.result.json`, `muse_decode_layer/hypotheses/done/neutral/combo-measured-rows-slide-batched-preattn-t8-tail.patch`; branch `hyp/muse_decode_layer/combo-measured-rows-slide-batched-preattn-t8-tail`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 118.05 | 118.48 | 1.0002 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 122.2 | 124.31 | 0.9999 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 195.64 | 210.54 | 0.9918 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 296.66 | 290.53 | 1.0182 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 239.18 | 239.48 | 1.0064 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 430.66 | 446.42 | 0.9949 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 180.39 | 181.08 | 1.0153 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 199.99 | 189.51 | 1.0327 | gained |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 427.64 | 414.69 | 1.0166 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 570.07 | 593.41 | 1.0339 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 606.99 | 599.76 | 0.9805 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 869.34 | 866.62 | 0.9949 | flat |

gained on full verify_1, verify_16. Expected hit: 3/8. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: combo-measured-rows-slide-batched-preattn-t8-tail`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

