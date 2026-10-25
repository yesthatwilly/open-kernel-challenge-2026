---
problem: muse_attention_decode
slug: full-decode1-sk-two-level-fused-preattn
created: 2026-10-25T054448Z
base: master
area: full-layer attention (batch-1 decode, stream-K prologue fusion)
change: >
  Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel:
  ~2.6 us of the row is the second graph node and its launch gap. PDL did not close it (plain
  27.74 vs PDL 28.29/31.58 us, c0f1945a108c4aa2) and cluster merges did not beat the two-level
  ticket (24.9 vs 25.0 us, f519261b7c2b435f). Lever: delete the pre-attention node at B=1, as
  stage 3 (full-decode-splitkv-s3-fused-preattn, 825baa03798f431a) did for 1<B<=64 (where it saved
  7.2 us at B=32). On master's kernel add the FUSED_PRE flag taken from the s3-fused-preattn
  branch and instantiate it with the two-level ticket (TWO=true). Split of the prologue work: q
  (16 heads x 64, RMSNorm + interleaved RoPE + e4m3, from the qkv row and norm weights, ~2 KB) is
  built by each CTA into its smem q slot (132 CTAs x 2 KB of L2 reads, negligible); the new K/V
  row (norm+rope K, FP8 K and V) is computed by the last-tile owner, stored to the cache AND used
  from registers/smem for its own last tile, so no other CTA reads it (only the last tile contains
  key seq-1). Keep the old-key gather unchanged. Self-check: cache rows bit-exact vs master's
  pre-attention kernel, output rel L2 vs fp32 reference at 80000/4097/64/63/1 keys, kv 1.0/0.25,
  grids 3/7/SM. Gate: interleaved graph replay of [fused two-level] vs [pre-attn + two-level] at
  B=1, route at >= 1.03x. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync,
  no allocation sized by input values (shape/config only), no data-dependent Python control flow,
  no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8,
  q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static
  global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer
  (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch,
  hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32
  reference with two states and alternating graph replays with refilled inputs (capture once,
  refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path
  for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs
  (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check
  launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256
  MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH
  CHOSEN prints both numbers.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "22.6 us shape: remove the 1.4 us pre-attention kernel and its gap; target shape <= 21 us (floor 4.3)"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: B>1 stays on master v2 (this run does not route v3 at B>1)
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: medium
depends_on: full-decode-splitkv-s3-fused-preattn
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>"
review: proceed
status: done
review_at: 2026-10-25T054716Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T054717Z
base_sha: dbb845684aa94ad1867281815e42f53464870f19
base_blob: 50274a3d8678d5656197b8e401ca411d4f08a6e2
branch: hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn
worktree: .worktrees/muse_attention_decode/full-decode1-sk-two-level-fused-preattn
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T084647Z
impl_commit: 2236339719c79707da71c91fd7a40228da43bdc9
submission_sha256: 39a86a5c348bd983ff953edfbfb59a9216acd0b46cc7a708005325af6f88e777
norm_sha256: e4d308f3c69ff0019f3a0d59da2a94f4268593f73e3bd84e11a63aa8f08162cb
submission_blob: fd1c193f0c8d523702cf3687711a06c01db9bce5
lint_warnings: 27
attempt_file: 2026-10-25T084648Z-check-full-decode1-sk-two-level-fused-preattn.md
prepared_at: 2026-10-25T084648Z
job: 2038a530edd04309
dispatched_at: 2026-10-25T084656Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.4410831181175277
speedup_ci: [1.4194668713457648, 1.4626993648892905]
harvested_at: 2026-10-25T091104Z
best_at_harvest: 1.444132502888963
promotion: null
---

# full-decode1-sk-two-level-fused-preattn

## Change

Full decode_1 is 22.6 us shape for 18.6 us of stream-K kernel + 1.4 us _pre_attention_kernel: ~2.6 us of the row is the second graph node and its launch gap. PDL did not close it (plain 27.74 vs PDL 28.29/31.58 us, c0f1945a108c4aa2) and cluster merges did not beat the two-level ticket (24.9 vs 25.0 us, f519261b7c2b435f). Lever: delete the pre-attention node at B=1, as stage 3 (full-decode-splitkv-s3-fused-preattn, 825baa03798f431a) did for 1<B<=64 (where it saved 7.2 us at B=32). On master's kernel add the FUSED_PRE flag taken from the s3-fused-preattn branch and instantiate it with the two-level ticket (TWO=true). Split of the prologue work: q (16 heads x 64, RMSNorm + interleaved RoPE + e4m3, from the qkv row and norm weights, ~2 KB) is built by each CTA into its smem q slot (132 CTAs x 2 KB of L2 reads, negligible); the new K/V row (norm+rope K, FP8 K and V) is computed by the last-tile owner, stored to the cache AND used from registers/smem for its own last tile, so no other CTA reads it (only the last tile contains key seq-1). Keep the old-key gather unchanged. Self-check: cache rows bit-exact vs master's pre-attention kernel, output rel L2 vs fp32 reference at 80000/4097/64/63/1 keys, kv 1.0/0.25, grids 3/7/SM. Gate: interleaved graph replay of [fused two-level] vs [pre-attn + two-level] at B=1, route at >= 1.03x. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (master's self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, hold no data and stay as they are). Every new CUDA route self-checks in prepare against an fp32 reference with two states and alternating graph replays with refilled inputs (capture once, refill A, replay, refill B, replay, compare each), and on any miss falls back to master's path for that band with a PATH CHOSEN line naming the failing case. Build as the landed TUs (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, in-process canary, sync+print after each self-check launch). Gates route a band only when the printed interleaved graph-replay timing (cold L2, 256 MB flush, empty-graph baseline subtracted) shows the new path winning by the stated margin; PATH CHOSEN prints both numbers.

## Expected

- `*layer=full load=decode_1 *` faster: 22.6 us shape: remove the 1.4 us pre-attention kernel and its gap; target shape <= 21 us (floor 4.3)
- `*layer=full load=decode_16 *` flat: B>1 stays on master v2 (this run does not route v3 at B>1)
- `*layer=sliding *` flat: untouched

## Risk

Stage 4 of the fused-preattn line (B=1 extension); builds on master code plus the s3-fused-preattn prologue. Risk: the last-tile owner's new-key build serialises a few hundred ns before its last tile; the per-CTA q build adds L2 reads to every CTA's prologue (v3 measured +3.1 us of build at B=16).

## Milestones

Rewrite stage 4 of `full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T054716Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage 4 that removes the 1.4 us _pre_attention_kernel at full decode B=1, and its in-process interleaved gate already gives an A/B. Two parts of the spec need fixing before the build: it claims about 2.6 us of node gap that the profiles do not show, and it asks for RoPE, which full layers do not use.

### Justification
Mechanism: at full decode_1 the shape is 22.6 us. The kernels are splitkv_attn_sk_kernel<4,true,true> at 18.6 us and _pre_attention_kernel at 1.4 us. Folding the pre-attention work into the stream-K kernel removes one graph node. The ownership argument holds: tile ranges are disjoint and key seq-1 is cache_loc[b], so exactly one CTA owns the new K/V row. The per-CTA q build is redundant but cheap in bytes (132 x 2 KB of L2). WEAKEST LINK, in two parts. (1) The gain is overstated. The spec attributes ~2.6 us of the row to 'the second graph node and its launch gap'. But sliding decode_1 runs ONE kernel and shows the same residual: 13.1 us shape vs 10.2 us kernel, 2.9 us. So the residual comes from each graph replay, not from each node (the judge already subtracts ~4 us of fixed replay cost). PDL from pre-attention also did not move decode_1 (23.08 vs 22.6 us), which fits that reading. The 7.2 us B=32 saving cited for s3-fused-preattn is not in its result either: shape 223.1 vs master 226.4 us, about 3.3 us, and at B=16 109.6 vs 111.1 us, about 1.5 us, with the fused kernel itself growing (the spec's own risk note says +3.1 us of build at B=16). Realistic ceiling: about 1.4 us, minus whatever the q build adds to the critical path. That gives roughly 21.2-21.8 us against the stated target of <=21 us, which needs everything to go right. Floor is 4.3 us, so M3 (<=1.5x floor) is out of reach and is not what this stage is for. (2) Latency of the q build. At B=1 every CTA has very little work (80K keys / 132 CTAs, about 9-10 tiles of 64). So a serialized load of qkv, then RMSNorm, then the FP8 cast before the first QK could add 0.5-1 us to every CTA and erase the saving. The s3 review already flagged this, and sliding-fused-hoist-first-tile showed that prologue ordering is fragile here. Build it so the first KV tiles' cp.async are issued BEFORE the q build, and the 2 KB qkv slice rides the ring with the first tile. SPEC ERROR: the change says 'RMSNorm + interleaved RoPE' for q and 'norm+rope K'. Full layers have NO RoPE (TASK.md lines 18/29; master full-layer pre-attention runs with ROPE=False). Mirror master: scaleless RMSNorm, eps 1e-5, then the e4m3 cast, no rotation. Otherwise the bit-exact cache self-check fails and correctness falls back. Expected directions are consistent: full decode_1 (weight 16) faster, decode_16/32 and sliding flat. The headline effect is about 16 x 1.4 / 2608 = 0.9%, inside the CI, which is acceptable for a rewrite judged on kernel time.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket: this is the incumbent kernel the change modifies, not an independent confounder. The prepare-time interleaved graph-replay gate ([fused two-level] vs [pre-attn + two-level]) is already an in-process A/B against it.
- fused-preattention-triton (plus paired-heads-launch-cleanup and its ab-tune batch gate): the fused prologue must reproduce master's B=1 pre-attention output exactly. At B=1 the paired predicate (batch > 1) is false, so q uses the unpaired layout that the two-level kernel reads today. The bit-exact cache-row check covers K/V, but q's smem layout and scale must also match what splitkv_attn_sk_kernel<4,true,true> consumes.
- full-decode-splitkv-s3-fused-preattn (depends_on, not landed, within-ci): FUSED_PRE was written against the s2 single-level splitkv_attn_sk_fused_kernel, not master's <4,true,true> template. Porting it is a merge, not a flag flip. Its measured net saving (about 1.5 us at B=16, about 3.3 us at B=32) is the honest prior, not 7.2 us.
- full-decode1-sk-pdl-from-preattn: a null PDL result at decode_1 suggests there is little inter-node gap to recover. That caps this change at about the pre-attention kernel's own 1.4 us.

### Interference and A/B needs
No companions. This is a rewrite stage gated by an in-process interleaved graph-replay A/B against exactly the path it replaces, with PATH CHOSEN printing both numbers. That alone makes a gain, a loss or a fallback readable from one run. The only overlapping landed change is the incumbent itself. Reverting the pre-attention or pairing landings would just break the comparison. With 8 checks and 8 queued hypotheses, a companion would buy nothing that the printed gate numbers do not already give.

### Open questions
- Where does the ~2.6-2.9 us shape-vs-kernel residual at B=1 come from, given that a single-kernel shape (sliding decode_1) shows the same amount? If it is per-graph cost, this stage's ceiling is about 1.4 us, and the next decode_1 lever is inside the kernel's 18.6 us (0.89 TB/s vs about 4.8 TB/s): more bytes in flight per CTA and the cost of the two-level merge tail.
- Does the q build overlap the first tiles' cp.async prologue? Print the eager per-kernel time of the fused kernel vs <4,true,true> in prepare, so the q-build cost can be separated from the node removal.
- When the gate does not route, does the run still record the fused kernel's correctness, so that stage 5 can build on it?


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn has no sync/.item(); allocations are shape-only (zero-element q placeholder); branch is config + qkv.data_ptr() alignment (host metadata), no input values.
- [x] Nothing derived from `data` cached: only zero-reset int32 ticket arrays (master's) stay global; no pointer/TMA/workspace globals.
- [x] n/a: no weights in this problem.
- [x] Activations/KV/q: KV FP8, q e4m3 only, fp32 softmax/merges, bf16 out.
- [x] Every key attended (stream-K covers all tiles; new key included from the smem copy).
- [x] No timing patches, harness access, subprocess or network (load_inline in-process, as master).
- [x] All work inside the timed call.
- [x] sm_90a: TORCH_CUDA_ARCH_LIST=9.0a around the build (master's _build_sk), no wgmma/setmaxnreg added, -lineinfo kept; per-launch prints in self-check; no spin waits added.

okc_lint: "OK with warnings" (only host-sync warnings in prepare-time code, outside kernel_fn). py_compile OK, __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (opus), against `master:muse_attention_decode/submission.py`. **Verdict: dispatch.** I made no edits.

- **No RoPE on full layers.** The fused route requires `not theta`. reference.py sets rope_theta 0.0 on full layers, and master already uses `ROPE=bool(theta)`, so the predicate matches. The kernel does a scaleless RMSNorm (with eps), then the e4m3 cast, and folds 5.473 into score_scale. That is exactly what `_pre_attention_kernel` does with ROPE=False.
- **Cache-row tolerance.** The self-check passes a new row only when at most 3 of its 128 bytes differ AND its rel L2 against master's row is below 2e-2. The rules' bound is CACHE_TOL = 8e-2 rel L2 against the reference, so the check sits well inside it. Bytes only differ at e4m3 rounding ties (Triton's approximate sqrt/div against IEEE), so this gate is not one that always fails.
- **New-row ownership is race-free.**
  - gx = min(grid, T), and CTAs at or above gx exit. So CTA gx-1 always ends at tile T-1 with N >= 1 and holds key seq-1.
  - The order is: the old row's cp.async lands, then a wait and a barrier, then the smem overwrite, then the global store at cache_loc, then one more barrier.
  - The swizzle `(kh*4+dd/16) ^ (tt&7)` matches issue_tile.
  - No other CTA reads row loc, and no pending cp.async targets that stage.
  - The k/v offsets in the qkv row (k at HQ*D, v at (HQ+HKV)*D) and the cache layout `(loc*HKV+kh)*D` match master.
- **q 16 B routing.** `qkv.data_ptr() % 16` reads pointer metadata that is fixed at capture. It is not a branch on tensor values. A misaligned pointer takes the pre-attention + two-level path.
- **Globals.** There are two: the import-time `_FUSED_B1` bool and master's self-resetting ticket arrays. No device pointer, TMA map or state-sized buffer is cached. The C++ `static bool done[5]` holds attribute flags only.
- **Two-state self-check.**
  - Two states with separate graphs, replayed 0 1 1 0 0 1 after a refill, with NaN-poisoned outputs and a ticket-sum check.
  - The canary and each case are synced and announced.
  - Any miss or raise keeps pre-attention + two-level, and PATH CHOSEN gives the reason.
- **kernel_fn.**
  - Branches only on config plus the alignment predicate.
  - Allocations depend on shape only (the q placeholder has zero elements).
  - No sync, print or try.
  - okc_lint: OK, with host-sync warnings only in prepare-time code.
  - py_compile: OK.
- **Call sites and q layout.** Every sk_attn call site was updated for the new cache_loc argument. The q fragment's column and byte mapping matches master's `qr` read.

Residual risk: nobody has compiled the CUDA yet, since this node has no nvcc. A compile error in the shared `_SK_CUDA` TU would also drop B>1 from v2 to v1. The run would print that loudly, but it would look like a regression. Beyond that, the gate may simply not reach 1.03x.

## Result

- job `2038a530edd04309` (check), status done, passed True, tests 16/16
- speedup 1.4411x, 95% CI [1.4195, 1.4627], score 2613.5203116666025 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.444132502888963, "ci": [1.4224705153456287, 1.4657944904322975], "job": "341623fa53d94cef"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-fused-preattn.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk-two-level-fused-preattn.patch`; branch `hyp/muse_attention_decode/full-decode1-sk-two-level-fused-preattn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.49 | 12.99 | 0.9983 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.59 | 21.6 | 1.2828 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.67 | 15.43 | 1.0053 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 109.55 | 110.8 | 0.9844 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.4 | 24.41 | 0.9911 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 223.11 | 227.6 | 0.9731 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.35 | 16.89 | 0.9979 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.04 | 29.89 | 1.006 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 23.23 | 22.7 | 1.0095 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.39 | 163.01 | 0.9926 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.6 | 28.93 | 1.0063 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 300.35 | 301.53 | 0.9904 | flat |

gained on full decode_1. Expected hit: 3/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 4 of `full decode batch 1: _pre_attention_kernel launch + master splitkv_attn_sk_kernel<4, two-level ticket>`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode1-sk-two-level-fused-preattn --stage 5`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-sk-two-level-fused-preattn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

