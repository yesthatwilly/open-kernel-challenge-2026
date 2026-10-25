# muse_attention_decode · 2026-10-25T084648Z · check · full-decode1-sk-two-level-fused-preattn

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T054448Z-full-decode1-sk-two-level-fused-preattn.md` (branch `hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn`).

- **Job:** `2038a530edd04309` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `22363397`

## Goal
Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel: ~2.6 us of the row is the second graph node and its launch gap. PDL did not close it (plain 27.74 vs PDL 28.29/31.58 us, c0f1945a108c4aa2) and cluster merges did not beat the two-level ticket (24.9 vs 25.0 us, f519261b7c2b435f). Lever: delete the pre-attention node at B=1, as stage 3 (full-decode-splitkv-s3-fused-preattn, 825baa03798f431a) did for 1<B<=64 (where it saved 7.2 us at B=32). On master's kernel add the FUSED_PRE flag taken from the s3-fused-preattn branch and instantiate it with the two-level ticket (TWO=true). Split of the prologue work: q (16 heads x 64, RMSNorm + interleaved RoPE + e4m3, from the qkv row and norm weights, ~2 KB) is built by each CTA into its smem q slot (132 CTAs x 2 KB of L2 reads, negligible); the new K/V row (norm+rope K, FP8 K and V) is computed by the last-tile owner, stored to the cache AND used from registers/smem for its own last tile, so no other CTA reads it (only the last tile contains key seq-1). Keep the old-key gather unchanged. Self-check: cache rows bit-exact vs master's pre-attention kernel, output rel L2 vs fp32 reference at 80000/4097/64/63/1 keys, kv 1.0/0.25, grids 3/7/SM. Gate: interleaved graph replay of [fused two-level] vs [pre-attn + two-level] at B=1, route at >= 1.03x. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## What changed, and where
Base: the claim was stale (worktree base dbb8456 = s2 + the unlanded s3-fused-preattn sibling; master bfd621c has s2 + the two-level ticket). I started from master's submission.py (the s3 separate-TU kernel is not carried: it is a sibling hypothesis and not on master) and ported the fused prologue into master's `splitkv_attn_sk_kernel` as a new template flag, `<4, true, true, FUSED=true>` = sk_attn variant 4. Non-fused instantiations compile to the same code (flag is constexpr); kc/vc became non-const pointers, `cache_loc` was added to `sk_attn` (all call sites updated). All edits are in muse_attention_decode/submission.py only.

What the fused kernel does (batch 1 only, TORCH_CHECK B == 1):
- The request's raw bf16 q row (2 KB) rides block 0's page-index cp.async group; the last CTA of the effective grid (it owns tile T-1, hence key seq-1) also brings the raw new k/v heads (512 B) and cache_loc[0].
- After tiles 0..2 are issued, every CTA builds its q MMA fragments in registers (scaleless RMSNorm eps 1e-5, then e4m3 cast, NO RoPE, no 5.473 factor), overlapping the first tiles' loads.
- The last CTA's warps 0-3 build the new row's e4m3 bytes (k: norm / k_scale, v: / v_scale) into smem during the same window.
- In the last tile's iteration (after the landing wait + barrier) the new bytes are written over that key's slot in the ring stage in smem, and stored to k/v cache at cache_loc; one extra __syncthreads once per launch. The cp.async read of the old row is discarded, so nothing reads the new row back through cp.async.
- Python: `_FUSED_B1` flag (set only by the import-time gate), `_full_decode_sk_fused`, `_sk_fused_case` / `_sk_fused_replay_check` / `_sk_fused_gate` / `_sk_validate_fused` / `_sk_fused_diagnostics`, kernel_fn routing (config-only plus qkv 16 B alignment, host metadata), PATH CHOSEN lines.

Review items:
- SPEC ERROR (no RoPE on full layers): applied; mirrors ROPE=False.
- Issue first KV tiles' cp.async before the q build: applied; q build is after the prologue tile issue (q even rides the index group, earlier than "with the first tile").
- Last-tile owner must not read the new row back via cp.async: applied (smem overwrite after the landed tile, global store after that read).
- Print eager per-kernel time of fused vs <4,true,true>: `_sk_fused_diagnostics` prints pre-attn, two-level and fused times and the differences.
- Gain is overstated (~1.4 us ceiling): gate demands >= 1.03x in interleaved graph replay (45 reps, cold L2, empty-graph baseline subtracted), both numbers printed.
- Bit-exact cache rows: not achievable (Triton approx sqrt/div vs IEEE); check allows <= 3 of 128 bytes to differ, rel L2 < 2e-2, other rows unchanged, cache_loc == page_table[seq-1] asserted. Documented deviation.
- q smem layout/scale vs what <4,true,true> consumes: fragments built with the identical column mapping; self-check compares output with the two-level kernel (< 5e-3).
- Open question 3 (record correctness when gate declines): case results and the gate timing are printed before the route decision, so stage 5 can build on it.
- Self-check: 80000/4097/64/63/1 keys, kv 1.0/0.25, grids 3/7/SM, canary; two states with separate graphs replayed alternately (0 1 1 0 0 1) with refilled inputs; ticket sums checked. Not run: no GPU/nvcc on this node.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn has no sync/.item(); allocations are shape-only (zero-element q placeholder); branch is config + qkv.data_ptr() alignment (host metadata), no input values.
- [x] Nothing derived from `data` cached: only zero-reset int32 ticket arrays (master's) stay global; no pointer/TMA/workspace globals.
- [x] n/a: no weights in this problem.
- [x] Activations/KV/q: KV FP8, q e4m3 only, fp32 softmax/merges, bf16 out.
- [x] Every key attended (stream-K covers all tiles; new key included from the smem copy).
- [x] No timing patches, harness access, subprocess or network (load_inline in-process, as master).
- [x] All work inside the timed call.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a around the build (master's _build_sk), no wgmma/setmaxnreg added, -lineinfo kept; per-launch prints in self-check; no spin waits added.

okc_lint: "OK with warnings" (only host-sync warnings in prepare-time code, outside kernel_fn). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Stage 4 of the fused-preattn line (B=1 extension); builds on master code plus the s3-fused-preattn prologue. Risk: the last-tile owner's new-key build serialises a few hundred ns before its last tile; the per-CTA q build adds L2 reads to every CTA's prologue (v3 measured +3.1 us of build at B=16).

## Result (after the run)
- job `2038a530edd04309` (check), status done, passed True, tests 16/16
- speedup 1.4411x, 95% CI [1.4195, 1.4627], score 2613.5203116666025 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-fused-preattn.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-fused-preattn.patch`; branch `hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn`
