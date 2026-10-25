---
problem: muse_decode_layer
slug: dense-chain-s3-t1-q-sk2-o-route-unpool
created: 2026-10-25T054152Z
base: master
area: "whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler)"
change: >
  Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2
  dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on an old base). Built on
  full-decode-sk-ticket-restack-b1-b64, whose two-level stream-K kernel replaces
  FA3+combine+pad/unpad+FusedRMSNormGate at full decode_1; the 04:36Z reviewer noted s2's
  _dl2_attention would otherwise preempt that path, so this stage COMPOSES them instead of
  choosing. Three pieces, each behind its own state flag, selector and fallback, all tokens == 1
  only (T>1 byte-identical to the parent): (A) full decode_1 attention side: dl_qkv_kernel<NF=1>
  (from branch hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate: qkv GEMV + q/k
  RMSNorm + RoPE + FP8 encode + KV store, measured 8.3 us, 4.5 TB/s) writes q in the unpaired
  layout splitkv_attn_sk_kernel<4,true,true> reads, then the sk two-level kernel (norm-gate
  epilogue, 20.1 us), then dl_o_kernel<NF=1> (6.5 us, PDL) into master's residual path; the s2
  measurement was custom Q+O 18.6 vs master attention side sans FA3 27.3 us. Sliding decode_1
  keeps master's _sliding_fused_kernel side (s2 was flat there). (B) from branch
  hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1: per-warp top-8 R with PDL after P (run
  87211e13 selector 87.3 vs 93.0 us summed, R 7.0 -> 5.8 us at T=1), both layers. (C) same branch:
  dl_unpool_kernel at NF=1 (T<=8 code, used here at T=1 only) replacing _gemm(expert_output,
  unpooler[:1024]) + F.pad (6.5 vs 9.2 us); boundaries stay on SGLang
  FusedPostNormResidualPreNorm. Those two runs were on a base without split-KV; hand-merge their
  _DL_CUDA TU edits into master's current dense-chain TU under one new module name, keeping
  master's w2 atomic-add Marlin flags at T=1 untouched. Self-check: Q/O at T=1 full vs master
  incl. KV cache bytes within one FP8 code, R on no-local / one-hot / near-tie, unpool rel L2 <=
  2e-3 with zero columns exactly 0, and the full T=1 chain under alternating-state replays. HARD
  RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except,
  allocation on input values, data-dependent control flow or per-call mutable state; gates read
  config only (layer, tokens/batch, q_len, page); nothing cached from inputs; precision per
  TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES:
  the judge prepares two state copies and rotates them, so tickets, workspaces, device pointers
  and descriptors live in the per-state dict built in prepare(), never in module/static globals.
  Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with
  refilled inputs and compare each to master; a miss turns off only the failing piece (fallback to
  master for it) and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only, print
  regs/smem/local/binaryVersion, canary launch + synchronize with launching/ok prints. Each
  selector prints CUDA-event timing new vs incumbent at T=1 and keeps the incumbent if new is
  slower (LESSONS 03:41Z). First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "parent ~134 us expected; Q/O replaces nvjet qkv+norm/rope/cast/store+nvjet o (-4.7..-8.7 us in s2), R -1.2, unpool -2.7: ~-8..-12 us -> ~123-126. M2: Q+sk2+O chain <= 36 us in prepare timing; kill: shape not 4 us below parent (z>=2)"
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "R + unpool only (stack run 87211e13: -3.7 us vs its master); 123.9 -> ~120.5. Kill: not below 122.5"
  - shapes: "*load=decode_16*"
    direction: flat
    why: T>1 byte-identical to the parent
  - shapes: "*load=decode_32*"
    direction: flat
    why: T>1 unchanged
  - shapes: "*load=verify_*"
    direction: flat
    why: "verify_1 is T=8: route/unpool gated to tokens == 1 here, so flat"
risk: medium
depends_on: full-decode-sk-ticket-restack-b1-b64
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: "T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)"
review: proceed
status: running
review_at: 2026-10-25T054418Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T054418Z
base_sha: 06cbcc08c8e651f4bb7e2791d8b1321bdb54ee3e
base_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
branch: hyp/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool
worktree: .worktrees/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool
mode: null
restack_of: null
stale_base: null
---

# dense-chain-s3-t1-q-sk2-o-route-unpool

## Change

Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2 dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on an old base). Built on full-decode-sk-ticket-restack-b1-b64, whose two-level stream-K kernel replaces FA3+combine+pad/unpad+FusedRMSNormGate at full decode_1; the 04:36Z reviewer noted s2's _dl2_attention would otherwise preempt that path, so this stage COMPOSES them instead of choosing. Three pieces, each behind its own state flag, selector and fallback, all tokens == 1 only (T>1 byte-identical to the parent): (A) full decode_1 attention side: dl_qkv_kernel<NF=1> (from branch hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl-t1-gate: qkv GEMV + q/k RMSNorm + RoPE + FP8 encode + KV store, measured 8.3 us, 4.5 TB/s) writes q in the unpaired layout splitkv_attn_sk_kernel<4,true,true> reads, then the sk two-level kernel (norm-gate epilogue, 20.1 us), then dl_o_kernel<NF=1> (6.5 us, PDL) into master's residual path; the s2 measurement was custom Q+O 18.6 vs master attention side sans FA3 27.3 us. Sliding decode_1 keeps master's _sliding_fused_kernel side (s2 was flat there). (B) from branch hyp/muse_decode_layer/stack-route-topk-pdl-unpool-nf1: per-warp top-8 R with PDL after P (run 87211e13 selector 87.3 vs 93.0 us summed, R 7.0 -> 5.8 us at T=1), both layers. (C) same branch: dl_unpool_kernel at NF=1 (T<=8 code, used here at T=1 only) replacing _gemm(expert_output, unpooler[:1024]) + F.pad (6.5 vs 9.2 us); boundaries stay on SGLang FusedPostNormResidualPreNorm. Those two runs were on a base without split-KV; hand-merge their _DL_CUDA TU edits into master's current dense-chain TU under one new module name, keeping master's w2 atomic-add Marlin flags at T=1 untouched. Self-check: Q/O at T=1 full vs master incl. KV cache bytes within one FP8 code, R on no-local / one-hot / near-tie, unpool rel L2 <= 2e-3 with zero columns exactly 0, and the full T=1 chain under alternating-state replays. HARD RULES: CUDA-graph capturable; kernel_fn has no host syncs, timing, prints, try/except, allocation on input values, data-dependent control flow or per-call mutable state; gates read config only (layer, tokens/batch, q_len, page); nothing cached from inputs; precision per TASK.md; every key attended; no child processes (build and self-check IN-PROCESS). TWO STATES: the judge prepares two state copies and rotates them, so tickets, workspaces, device pointers and descriptors live in the per-state dict built in prepare(), never in module/static globals. Self-check: capture once per state, then alternate graph replays of state 0 and state 1 with refilled inputs and compare each to master; a miss turns off only the failing piece (fallback to master for it) and prints PATH CHOSEN ... (fallback: <reason>). Build sm_90a only, print regs/smem/local/binaryVersion, canary launch + synchronize with launching/ok prints. Each selector prints CUDA-event timing new vs incumbent at T=1 and keeps the incumbent if new is slower (LESSONS 03:41Z). First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: parent ~134 us expected; Q/O replaces nvjet qkv+norm/rope/cast/store+nvjet o (-4.7..-8.7 us in s2), R -1.2, unpool -2.7: ~-8..-12 us -> ~123-126. M2: Q+sk2+O chain <= 36 us in prepare timing; kill: shape not 4 us below parent (z>=2)
- `*layer=sliding load=decode_1 *` faster: R + unpool only (stack run 87211e13: -3.7 us vs its master); 123.9 -> ~120.5. Kill: not below 122.5
- `*load=decode_16*` flat: T>1 byte-identical to the parent
- `*load=decode_32*` flat: T>1 unchanged
- `*load=verify_*` flat: verify_1 is T=8: route/unpool gated to tokens == 1 here, so flat

## Risk

Three pieces in one stage are attributable by their prepare selector lines and by the sliding decode_1 row (B+C only) vs the full decode_1 row (A+B+C). If the parent did not land, build on its branch anyway (rewrite rule). Composition risk: q layout and FP8 cache encode between dl_qkv_kernel and the sk kernel.

## Milestones

Rewrite stage 3 of `T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054418Z by an adversarial subagent (model opus).

**Verdict: proceed.** This rewrite stage combines pieces that were each measured separately (s2 Q/O, the warp top-k R and the NF=1 unpool) on top of the not-yet-run sk two-level parent. The design can reach its targets, but the full decode_1 estimate counts the o_norm_gate removal twice, and the sliding decode_1 kill line sits inside run-to-run drift. Proceed with no companion runs, and attribute through the per-piece prepare-time selector lines.

### Justification
The mechanism is plausible. At T=1 the replaced work is bound by weight reads and launch count. s2's dl_qkv_kernel streamed the 2304x8192 qkv_gate at 4.5 TB/s (8.3 us), and nvjet qkv + qk_norm + FP8 casts + store_cache are separate launches on master (kernel_fn lines ~2475-2495). The s2 run c6cb53d8 moved full decode_1 142.93 -> 137.98 (+4.65%, gained), and stack-route-topk-pdl-unpool-nf1 (87211e13) moved sliding decode_1 127.48 -> 122.44 and full decode_1 142.93 -> 136.93. Current master is 123.9 / 139.2 us against floors of 40.0 / 44.2, so neither row is near its floor. WEAKEST LINK: the full decode_1 estimate (-8..-12 us -> 123-126) double-counts. s2's 'custom Q+O 18.6 vs master sans FA3 27.3 us' included dl_o_kernel's prologue 1 absorbing the unpad copies and o_norm_gate (FusedRMSNormGate). The parent's sk two-level kernel already removes FusedRMSNormGate (norm-gate epilogue), so in s3 dl_o runs with prologue 0 and that part of s2's 8.7 us saving is already in the parent's ~134. A realistic A saving vs the parent is ~4-6 us, plus R ~1.2 and unpool ~2.7, so expect ~126-128, not 123-126. The 'not 4 us below parent' kill line is still reachable. Second weak point: the sliding decode_1 row is the only clean B+C attribution, and its kill line (below 122.5, i.e. -1.1% vs 123.9) is inside the 1.5% noise band. Same-code rows drift by 2-4 us between runs (boundaries-unpooler-pdl-small-t, flat, read 123.87 on a 127.48 base; current master reads 123.9). The 87211e13 gain was measured against a 127.48 base, and some of that has since been absorbed by master (w2 atomic-add at T=1 also shortens the decode_1 rows). B+C may come in at ~2-3 us, below the significance threshold. The expected rows for decode_16/32 and verify (flat) match the tokens==1 gating: verify_1 is T=8 and correctly excluded. The headline effect is small (~(16*7 + 46*3)/17775 ≈ 1.4%), but as a rewrite this stage is judged on the replaced kernels' own time. Profile top_kernels at both decode_1 rows (dl_qkv, sk2, dl_o, R, unpool vs nvjet/qk_norm/store_cache/R/nvjet+pad) plus the selector CUDA-event lines make the milestones readable.

### Confounders on the current kernel
- full-decode-sk-ticket-restack-b1-b64 (parent, not yet run): A's saving is measured against whatever the parent's B=1 path ends up being. If the parent's B=1 selector keeps FA3 (sk slower or self-check miss), A must fall back to s2's FA3 + dl_o prologue-1 path, a different composition from the one specified. A full decode_1 gain would then mix 'sk2 absent' with 'Q/O present'. The q layout and gate source also change: dl_qkv writes a contiguous bf16 gate [T,1024], while the parent's norm-gate epilogue reads the gate out of qkv at qkv.stride(0). The sk kernel must be given gate_stride=1024, or a silent wrong-gate read results.
- w2-marlin-atomic-add-t1-port: its use_atomic_add path is gated on 'align is not None and dispatch is None' at num_tokens==1, and state['w2_mode'] is chosen by a prepare self-check that runs through kernel R's align. Piece B replaces R. If _w2_atomic_self_check still exercises the old R, or the new R's sorted_ids/n_padded padding differs (e.g. sentinel values for remote -1 slots), w2 can silently switch mode or mis-zero C. Any decode_1 move could then come from the w2 mode flipping rather than from R. Re-run the w2 self-check after the new R is installed, and print w2_mode in the PATH CHOSEN line.
- small-t-launch-minimal-dense-chain-pdl: B adds PDL after P in the P->R chain, and A adds PDL edges qkv->sk2->dl_o into SGLang's non-PDL FusedPostNormResidualPreNorm. PDL overlap savings depend on the neighbouring kernels' prologues, so R's -1.2 us measured on a base without the sk kernel need not transfer. This affects magnitude only, not the sign.
- post-expert-norm-triton-small-t: the unpool kernel (C) reads _post_norm_combine_kernel's bf16 output. The layout is unchanged, so this is not a real confounder beyond the unpool's own rel-L2 check.
- dense-chain-stage2-attn-side-pdl-t1-gate / stack-route-topk-pdl-unpool-nf1 (sources, not landed): their measured gains were on bases without split-KV and without w2 atomic-add (base 127.48/142.93 vs master 123.9/139.2). Part of their reported delta is therefore base-dependent and already absorbed.

### Interference and A/B needs
No companion runs are warranted. This is a rewrite stage with three pieces behind separate state flags, and each piece prints a prepare-time CUDA-event selector line (new vs incumbent) plus a PATH CHOSEN line. That gives per-piece attribution inside the single run. The row structure also separates them: sliding decode_1 = B+C, full decode_1 = A+B+C. The one real ambiguity is that the sliding row's B+C effect (~2-4 us) sits at the noise floor. A control for it would be a plain repeat of the already-run 87211e13 code on master, and identical code is never re-run. An ab-revert of w2-marlin-atomic-add-t1-port or of the parent would spend one of the 3 remaining decode checks on a question the selector logs and the self-check already answer. The next stage is the better place to isolate any piece that turns out ambiguous.

### Open questions
- If the parent's B=1 sk selector keeps FA3 (or the parent fails), does piece A fall back to s2's FA3 + dl_o prologue-1 composition, or turn off entirely? The change text specifies only the sk2 composition.
- Does the sk two-level kernel accept a separate contiguous gate tensor (stride 1024), or does it hard-code reading the gate from qkv at qkv.stride(0)? dl_qkv writes no bf16 qkv at NF=1 except in 'raw' mode.
- The dl_o 6.5 us figure: was it measured with prologue 1 (s2) or prologue 0? At prologue 0, after sk2, the expected saving vs the parent's nvjet o GEMM is the right comparison, not the s2 figure.
- Does _w2_atomic_self_check run after the new R is installed, and does the new R emit the same sorted_ids sentinel/padding that Marlin's atomic-add w2 assumes?
- Is the sliding decode_1 kill line (122.5) achievable given that 87211e13's -5 us was against a 127.48 base and current master already reads 123.9? Consider judging B+C on their profile kernel times (R 7.0 -> 5.8, unpool+pad 9.2 -> 6.5) rather than the row median.

