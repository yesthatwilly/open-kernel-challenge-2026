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
stage: dispatched
claimed_at: 2026-10-25T054418Z
base_sha: 06cbcc08c8e651f4bb7e2791d8b1321bdb54ee3e
base_blob: 62c5a9413ba5710defea2ca3de14c1a1dc3b5a6d
branch: hyp/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool
worktree: .worktrees/muse_decode_layer/dense-chain-s3-t1-q-sk2-o-route-unpool
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T091047Z
impl_commit: e4b884143211d9afb63a4a2d196a6aae1ea5fb81
submission_sha256: 887c43be53ff776e2cb33f103a9855d13c044a3d6e4c1276f45351a3bbfdc75f
norm_sha256: 44a5db33f1742645fa9b8d5413de5c629f9da15d24352adc0dafcf3a65ed0e39
submission_blob: 07cda658e21f2eb25a9dd1f50ac5202ad5985831
lint_warnings: 57
attempt_file: 2026-10-25T091047Z-check-dense-chain-s3-t1-q-sk2-o-route-unpool.md
prepared_at: 2026-10-25T091047Z
job: 34a9be06918d448b
dispatched_at: 2026-10-25T091101Z
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


## Implementation notes

Built 2026-10-25 by a Sonnet build agent. Not run (no GPU on the build node): every kernel edit below is a verbatim port from a branch that already ran on the judge, plus new glue; a g++ `-fsyntax-only` pass over both CUDA TUs with stub headers is clean, `py_compile` and okc_lint pass.

**Base.** The worktree was claimed before its parent existed. Step 1 replaced `muse_decode_layer/submission.py` with `git show hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64:muse_decode_layer/submission.py` (3709 lines), then the stage-3 change went on top (diff vs the parent: +1881/-90).

**What changed (all in `muse_decode_layer/submission.py`).**
- One new module name, `muse_decode_dense_chain_s3` (the stage-1 dense-chain TU, hand-merged), built at import with `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` (restored after), flags `-O3 -std=c++20 -lineinfo`, arch flags printed. No wgmma / setmaxnreg anywhere, so no `__CUDA_ARCH_FEAT_SM90_ALL` guard is needed. Stage 1's R is kept as `dl_route_norm_align_v1_kernel` (variant 2: every T > 1, and T = 1 when piece B is off, so T > 1 is the parent's code).
- From `hyp/.../stack-route-topk-pdl-unpool-nf1`: the per-warp top-8 kernel R (`dl_route_norm_align_kernel<true>`, griddepcontrol.wait after the constants, binary-search pad), `launch_r`, the `(pdl, variant)` wrapper, `griddep_launch()` at the end of P's main K loop (a no-op for a dependent launched without the attribute), `dl_unpool_kernel` NF=1 (`launch_u<1, 8, 1024>`, T <= 8 code, used at T = 1 only) and its `dl_unpool` wrapper. The `_DLB` boundary kernels of that branch are NOT ported (boundaries stay SGLang's).
- From `hyp/.../dense-chain-stage2-attn-side-pdl-t1-gate`: `dl_qkv_kernel<NF>` and `dl_o_kernel<NF>` with their launchers and the `dl2_qkv` / `dl2_o` wrappers; used here at NF = 1, mode 0 (unpaired q), prologue 0 only.
- sk kernel (`muse_decode_sk`): `finalize_store` no longer hard-codes the gate offset. `sk_attn` takes `gate_off`; the host passes `qkv.data_ptr() + gate_off` with the row stride, and TORCH_CHECKs that `gate_off + (B-1) * stride + 1024 <= numel` (a stride / offset mismatch throws, never reads another tensor's memory). `_full_decode_sk(..., gate=None)` keeps the parent's call (qkv, offset 1280, `qkv.stride(0)`); piece A passes `gate=` dl_qkv's contiguous [T, 1024] bf16 gate (offset 0, stride `gate.stride(0)` = 1024). The kernel also issues `griddepcontrol.launch_dependents` at each CTA's last tile so dl_o (the only dependent launched with the programmatic attribute) can prefetch its weights during the sk merge; a no-op for every other dependent.
- Python: `kernel_fn` is split (`_parent_attention` = the parent's attention side, verbatim); piece A (`_dl3_attention`) runs only when `state["dl3"]["a"]`, `tokens == 1` and `_sk_variant(cfg, state["sk"]) == 3` (config + the state's own flags). Piece B is `state["dl"]["r_new"]` (read by `_dl_forward` at tokens == 1, PDL after P). Piece C is `state["dl3"]["u"]` (tokens == 1 and the T=1 dense chain on). Sliding decode_1 keeps `_sliding_forward` and gets B + C only. Tickets, workspaces and flags: `state["dl3"]` (Q/O/unpool), `state["dl"]` (P/R, existing), `state["sk"]` (existing), all built per prepare() call; nothing device-side in module globals (the C++ statics are function-attribute bools only).
- prepare() order: `_dl_setup` -> `_dl3_setup_bc` (pieces B, C: kernel self-check + selector) -> align-parallel -> `_w2_atomic_self_check` -> split-KV -> `_sk_prepare` -> `_dl3_finish` (piece A, layer-level checks, PATH CHOSEN lines).

**Adversarial-review items, each applied.**
1. sk needs `gate_stride` for dl_qkv's contiguous gate (no silent wrong-gate read): done as above, plus a prepare check that the separate-gate sk output is bit-identical to the qkv-gate output (n = 4097 and 30000, kv_scale 1 and 0.25) and `gate.stride(0) == 1024`.
2. Parent's B=1 selector keeps FA3 (or sk is off for the state): piece A is OFF (reason printed), it never falls back to s2's FA3 + dl_o prologue-1 composition; B and C are independent of it. (Answers the first open question: turn off.)
3. dl_o runs with prologue 0 after sk (the s2 6.5 us figure was prologue 1 = unpad + o_norm_gate absorbed; the selector times the whole chain against the parent's actual chain, which is the right comparison; expected A saving ~4-6 us as the review says, the spec's 8-12 us double-counts o_norm_gate). The chain time (Q + sk + O) is printed against the 36 us M2 line.
4. w2: piece B is installed before `_w2_atomic_self_check`, whose `_dl_route` call now takes the new R at T=1 (so the self-check runs through the R in use); R's outputs (including sorted_ids sentinel / padding, block_experts, n_padded) are verified bit-identical to stage 1's R before B can turn on. If B is later turned off by a layer-level check the w2 self-check is re-run. `w2_mode` is printed in every PATH CHOSEN line.
5. dl_qkv KV store vs `store_cache`: the kernel reads `k_scale` / `v_scale` from the device tensors and divides in fp32 then rounds through bf16 (as `bf16 / 0-dim fp32` does); prepare compares q, k, v FP8 bytes to the parent's ops with a code-gap <= 1 gate (plus rel L2 <= 4e-2), untouched cache rows byte-equal, for kv_scale 1.0 and 0.25 (inputs scaled so v / scale stays inside E4M3, where torch's cast would give NaN and ours saturates).
6. Tickets / workspaces per state and self-resetting: yes (see above); the layer-level check builds a second state over the same constants with its own P/R, Q/O/unpool and sk buffers, captures one graph per state of the whole `kernel_fn` at T=1 (full kv_scale 0.25, and sliding), alternates six replays between them with refilled inputs (outputs poisoned first), compares each to the parent path run eagerly, and requires every ticket back at 0 after each replay.
7. Near-tie top-k vs SGLang's tie order: R's key is (biased score, lower expert id first). The check requires the new R bit-identical to stage 1's R (which the judge accepted) on random logits, no-local, one-hot, a 1e-7 near tie, and two EXACT-tie cases (zero router bias, 12 equal logits straddling the 8th place; two tie groups), with and without PDL; it also runs a zero-bias `FusedMuseSparkRouting` on the exact-tie cases and PRINTS whether its ids / order match ("tie order ..." lines). That comparison is information only: SGLang's own tie order is not documented and the judge's routing check allows a 0.05 score window.
8. Per-piece selector lines: `selector A/B/C` (CUDA-event graph replays behind a 256 MB flush, interleaved, median of 15, new must win by 1.02x) and `PATH CHOSEN: dl3 piece A/B/C ...` lines, each with `w2_mode`.
9. Open question on the sliding kill line (122.5 inside the noise band): not a build item; judge B + C on the profile rows (R 7.0 -> 5.8, unpool + pad 9.2 -> 6.5) and the `selector B` / `selector C` lines.

**Deviations from the spec.** (a) Self-check bound for q / k / v is stated as an FP8 code gap <= 1 (plus the s2 audit's rel L2 4e-2), not "within one code" for every element of every tensor at once: q and k go through RMSNorm + RoPE in fp32 then bf16, FusedQKNorm may round differently inside. (b) The layer-level hidden / stream bounds are 5e-2 / 2e-2 with piece A on (1e-2 / 5e-3 without) and the expert ids may differ by one swapped near-tie expert with A on (A off: equal). (c) The unused linear-pad instantiation of the new R from the stack branch is dropped; only the binary-search variant is compiled. (d) Sliding decode_1 does not use dl_qkv (s2 was flat there).

**Bisect on failure.** If the combined layer-level check fails, each piece is tried alone and only the failing ones are turned off (reason printed); if the survivors fail together, they are all turned off.

## Rules check

- [x] CUDA-graph capturable: no `.item()` / `.cpu()` / `synchronize()` / prints in `kernel_fn`'s call graph (okc_lint: every host-sync warning is outside kernel_fn, in prepare()); kernel_fn allocates only shape-sized outputs, branches on config, `tokens` and per-state flags set in prepare(); no try/except, no timing, no per-call mutable state. The layer-level check captures `kernel_fn` itself on two states.
- [x] Nothing derived from `data` is cached across calls: tickets / workspaces / flags are prepare()-time per-state buffers and booleans; scales, positions, cache_loc, page table are read from the device tensors by the kernels every call.
- [x] Weights: unchanged. No repack, no new precision: qkv_gate / o / unpooler are the same bf16 tensors in `state["dense"]`, read directly by the new kernels.
- [x] Activations BF16 where SGLang keeps BF16 (qkv GEMV outputs rounded through bf16 before norm / cast, gate bf16, o out bf16, unpool out bf16); KV cache FP8 E4M3 (k, v stored by dl_qkv_kernel as the parent's store_cache does); queries E4M3 only (q rounded via bf16 to E4M3, as FA3's path).
- [x] Attention sees every key: the sk two-level kernel is unchanged apart from the gate pointer and a launch_dependents; no top-k, sparse, block-skip or weight-based skipping.
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network: grep for subprocess / multiprocessing / os.fork / Popen / os.system finds nothing; the only builds are `load_inline` (as the parent's), in-process; checks run in-process in prepare().
- [x] All work happens inside the timed call; no output caching or replay (prepare()-only checks and selectors never feed outputs to kernel_fn).
- [x] Two states (judge `copies: 2`): n/a to globals, none added; the per-state check is described above.
- [x] sm_90a: n/a for wgmma (none used); the build is sm_90a-only via TORCH_CUDA_ARCH_LIST=9.0a, `-lineinfo`, cubin version 90 asserted per kernel in prepare(), regs / local / smem / CTAs-per-SM printed (`dl3_attrs`).
- [x] Self-check launches announced with a flushed print before and synchronize + print after (`_dl3_launch`); no spin-waits added (the new kernels use tickets only, no waits on other CTAs).

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`): `OK with warnings`: 57 `host-sync` warnings, all marked "outside kernel_fn's call graph; fine in prepare()", no errors, no other warning class. `python3 -I -m py_compile` clean; `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 (opus) against the parent `hyp/muse_decode_layer/full-decode-sk-ticket-restack-b1-b64`. **Verdict: dispatch-with-fixes.**

Fixes (applied in the worktree's `muse_decode_layer/submission.py` by a Python read/write script; py_compile, okc_lint (OK, 57 host-sync warnings, all outside kernel_fn) and a symtable undefined-name scan are clean; `__pycache__` removed):
1. **The build is now independent (top issue).** The combined module `muse_decode_dense_chain_s3` replaced stage 1's module, so a compile failure would have turned the landed dense chain off at every T. The parent's dense-chain TU and its build function are now kept verbatim (`_DL1_CPP` / `_DL1_CUDA` / `_build_dense_chain_v1`, module `muse_decode_dense_chain_v1`, `-O3 -arch=sm_90a`). They are built only if the s3 build fails. `_DL_S3` (set at import, from the build result only) routes `_dl_route_launch` to v1's 14-argument `dl_route_norm_align`, and `_dl3_setup_bc` turns pieces A, B and C off. In that case every T runs exactly the parent's stage-1 chain.
2. **The sk module gets the same treatment.** Stage 3 also edited the parent's sk TU (the `gate_off` argument and `launch_dependents`), so an sk build failure would have dropped the parent's sk paths at every batch. The s3 sk module is now named `muse_decode_sk_s3`. If it fails, the parent's exact TU is rebuilt as `muse_decode_sk`. That TU is derived by reversing 7 exact string edits (`_SK1_REVERSE`), and I checked offline that the reversal reproduces the parent's `_SK_CPP` / `_SK_CUDA` byte for byte. `_full_decode_sk` then makes the parent's call, and piece A is off (`_SK_S3`).

Checked with no change needed:
- The gate_off / gate_stride host TORCH_CHECK holds for the parent's qkv layout: 1280 + (B-1)·2304 + 1024 = B·2304. It also holds for dl_qkv's contiguous gate (0 + 1024 = 1024).
- The w2 self-check runs after piece B is installed, and goes through `_dl_route` with variant None, i.e. the R in use. It is re-run if B is turned off later, and it never raises.
- `k_scale` / `v_scale` are read in-kernel via `__ldg` from the device tensors. Only the `scaled` flag comes from config, as in the parent.
- Tickets and workspaces are per state (`_dl3_buffers` in prepare). Their sizes match the CUDA constants: Q 6·32·2304, O 2·32·8192, U = `WS_FLOATS`.
- The new R uses the same arithmetic and ordering as v1, so the bit-identical gate can pass.
- Every self-check, selector and layer check sits inside try/except, and a miss lands on the parent path with a PATH CHOSEN line.
- kernel_fn branches only on config and per-state flags. It has no syncs, try/except or prints, and allocates only shape-sized buffers.
- No undefined names in any f-string or except branch.

Residual risk: piece A's composition (dl_qkv's q / FP8 KV versus FusedQKNorm + store_cache, then PDL into dl_o) has never run on hardware. A self-check miss there only turns A off, but a device fault in a new kernel during prepare would still poison the context.
