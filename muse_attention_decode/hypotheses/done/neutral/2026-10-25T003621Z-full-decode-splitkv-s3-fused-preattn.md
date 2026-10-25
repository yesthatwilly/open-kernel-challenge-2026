---
problem: muse_attention_decode
slug: full-decode-splitkv-s3-fused-preattn
created: 2026-10-25T003621Z
base: master
area: full-layer attention (decode, splitkv stream-K kernel prologue)
change: >
  Stage 3 of the full-decode rewrite, built on the stage-2 branch
  hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token
  tiles + per-request ticket combine/norm-gate in splitkv_attn_sk_kernel, module
  muse_full_decode_splitkv_v2), whatever its score. Gate unchanged (_splitkv_gate: full, q_len 1,
  1 < batch <= 64, tp8, head_dim 64, page 1). One change: remove the separate
  _pre_attention_kernel launch on the gated path so full decode_16/32 run ONE kernel
  (REWRITE_READ_2026-10-25 sections 1 and 5 list fused pre-attention as the idea worth keeping on
  this kernel). New kernel splitkv_attn_sk_fused_kernel in a NEW module
  muse_full_decode_splitkv_v3 (v2 stays compiled as the fallback): takes the raw qkv
  pointer/stride, positions, cache_loc, inv_freq, k_scale, v_scale instead of the precomputed FP8
  q. (a) Q: when a CTA starts a request segment it builds that request q block in smem from qkv:
  the exact math of master _pre_attention_kernel for the 16 q heads (_norm_rope: scaleless RMSNorm
  over 64, eps 1e-5, RoPE with the same inv_freq table at positions[b], then the same FP8 cast and
  scale master writes), one warp per 2 heads, fp32 math; the redundancy (each request is touched
  by ~2-17 CTAs) is 2 KB of qkv per segment. (b) New K/V row: the CTA whose range contains request
  b last tile (key seq_b - 1, which is the token being decoded, stored at cache_loc[b]) computes
  the 2 kv heads new k (norm+rope+fp8) and v (fp8) exactly as master, stores them to
  k_cache/v_cache at cache_loc[b] (the cache update stays an output), then __syncthreads() before
  issuing that tile cp.async so the ring reads the stored row (same CTA, L2-coherent .cg loads).
  No other CTA reads key seq_b - 1, so no cross-CTA ordering is needed; the self-check asserts
  that ownership invariant. Launches on full decode_16/32 go from two (pre-attn, A) to one.
  BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image
  check + synthetic canary + self-check with torch.cuda.synchronize() and a print after each
  kernel; never a child process). The self-check additionally compares the stored cache rows of v3
  against master _pre_attention_kernel by rel L2 (not byte equality: LESSONS 2026-10-25, Triton
  sin/cos approximations) and q-dependent outputs by rel L2 < 5e-3. Fallback on any failure: v2
  path (pre-attn + v2 kernel), printing PATH CHOSEN: full decode streamk v2 (fallback: <reason>);
  success prints PATH CHOSEN: full decode streamk fused-preattn v3. Every other shape byte for
  byte. First measurement is an okc check, never a scored submit.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced: _pre_attention_kernel 2.1 us + its launch gap on top of the stage-2 kernel; M1 16/16 tests and <= stage-2 shape time; M2 shape -2..-3 us vs stage 2 (floor 63.2)"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced: _pre_attention_kernel 2.0 us + launch gap; M2 shape -2..-3 us vs stage 2 (floor 126.4)"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside the gate, master code byte for byte
risk: medium
depends_on: full-decode-splitkv-s2-streamk-ticket-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: _pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)
review: proceed
status: done
review_at: 2026-10-25T014055Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T014055Z
base_sha: f291b49c49670595d0615e749294f69bc15de1a4
base_blob: 695c97d2e9fef8f806ec02d2fe14a02668989e3f
branch: hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn
worktree: .worktrees/muse_attention_decode/full-decode-splitkv-s3-fused-preattn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T034225Z
impl_commit: dbb845684aa94ad1867281815e42f53464870f19
submission_sha256: b3dc55fa26451719bee99e40ab0f68e21f31286baad2e829ac91327bf871ef2f
norm_sha256: 1fd1af6913adfa2d5c5eeb7bcd11699de723fcbfda56d510f875849fb74daeba
submission_blob: 50274a3d8678d5656197b8e401ca411d4f08a6e2
lint_warnings: 16
attempt_file: 2026-10-25T034225Z-check-full-decode-splitkv-s3-fused-preattn.md
prepared_at: 2026-10-25T034225Z
job: 825baa03798f431a
dispatched_at: 2026-10-25T034233Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.42828155853798
speedup_ci: [1.4068573351599103, 1.4497057819160495]
harvested_at: 2026-10-25T043353Z
best_at_harvest: 1.4321773019894537
promotion: null
---

# full-decode-splitkv-s3-fused-preattn

## Change

Stage 3 of the full-decode rewrite, built on the stage-2 branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (stream-K over 64-token tiles + per-request ticket combine/norm-gate in splitkv_attn_sk_kernel, module muse_full_decode_splitkv_v2), whatever its score. Gate unchanged (_splitkv_gate: full, q_len 1, 1 < batch <= 64, tp8, head_dim 64, page 1). One change: remove the separate _pre_attention_kernel launch on the gated path so full decode_16/32 run ONE kernel (REWRITE_READ_2026-10-25 sections 1 and 5 list fused pre-attention as the idea worth keeping on this kernel). New kernel splitkv_attn_sk_fused_kernel in a NEW module muse_full_decode_splitkv_v3 (v2 stays compiled as the fallback): takes the raw qkv pointer/stride, positions, cache_loc, inv_freq, k_scale, v_scale instead of the precomputed FP8 q. (a) Q: when a CTA starts a request segment it builds that request q block in smem from qkv: the exact math of master _pre_attention_kernel for the 16 q heads (_norm_rope: scaleless RMSNorm over 64, eps 1e-5, RoPE with the same inv_freq table at positions[b], then the same FP8 cast and scale master writes), one warp per 2 heads, fp32 math; the redundancy (each request is touched by ~2-17 CTAs) is 2 KB of qkv per segment. (b) New K/V row: the CTA whose range contains request b last tile (key seq_b - 1, which is the token being decoded, stored at cache_loc[b]) computes the 2 kv heads new k (norm+rope+fp8) and v (fp8) exactly as master, stores them to k_cache/v_cache at cache_loc[b] (the cache update stays an output), then __syncthreads() before issuing that tile cp.async so the ring reads the stored row (same CTA, L2-coherent .cg loads). No other CTA reads key seq_b - 1, so no cross-CTA ordering is needed; the self-check asserts that ownership invariant. Launches on full decode_16/32 go from two (pre-attn, A) to one. BUILD/SAFETY exactly as stage 2 (TORCH_CUDA_ARCH_LIST=9.0a only, -lineinfo, in-process image check + synthetic canary + self-check with torch.cuda.synchronize() and a print after each kernel; never a child process). The self-check additionally compares the stored cache rows of v3 against master _pre_attention_kernel by rel L2 (not byte equality: LESSONS 2026-10-25, Triton sin/cos approximations) and q-dependent outputs by rel L2 < 5e-3. Fallback on any failure: v2 path (pre-attn + v2 kernel), printing PATH CHOSEN: full decode streamk v2 (fallback: <reason>); success prints PATH CHOSEN: full decode streamk fused-preattn v3. Every other shape byte for byte. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=full load=decode_16*` faster: replaced: _pre_attention_kernel 2.1 us + its launch gap on top of the stage-2 kernel; M1 16/16 tests and <= stage-2 shape time; M2 shape -2..-3 us vs stage 2 (floor 63.2)
- `*layer=full load=decode_32*` faster: replaced: _pre_attention_kernel 2.0 us + launch gap; M2 shape -2..-3 us vs stage 2 (floor 126.4)
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside the gate, master code byte for byte

## Risk

Small absolute win (~2-3 us on 120-250 us shapes) but it completes the one-launch full-decode path; risk is the new-key ownership and Q recomputation correctness, covered by the self-check. If stage 2 fails its gate, this branch still builds on its code and the v1 fallback.

## Milestones

Rewrite stage 3 of `_pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T014055Z by an adversarial subagent (model opus).

**Verdict: proceed.** Builds on a real stage-2 gain and is rules-clean, but the prize is small (~2 us of 111/227 us, ~2%) and the spec wrongly says to apply RoPE on full layers, which have none (rope_theta 0, master runs _pre_attention_kernel with ROPE=False); the builder must mirror master's ROPE=False math or the self-check will fail and the run will measure the v2 fallback.

### Justification
Stage 2 (job e592b1f5d3db4f83, 1.4171 [1.396,1.443] vs best 1.3949 [1.374,1.416], unproven-vs-best only because the CIs overlap) made a real, attributable kernel gain: full decode_16 shape 122.8 -> 110.9 us (A+B 120.0 -> splitkv_attn_sk_kernel 108.9), full decode_32 241.9 -> 226.8 us (A+B 254.3 -> 226.8); every other shape within ~1 us of best. M1 met on both; M2 narrowly missed (108.9 vs <=108, 226.8 vs <=222). The in-process diagnostics split it: stream-K work split alone gave 1.10x/1.14x on kernel A (117.5 -> 106.9, 245.3 -> 215.1 us eager), while the ticket fold saved only 1.05x/1.03x net: the ticket kernel is ~5.8/6.2 us slower than the no-ticket kernel, i.e. the serialized last-arriver merge tail costs ~6 us against the 11.7 us separate combine it replaces. A: the mechanism is sound but small. On the gated shapes _pre_attention_kernel is 2.2 us (decode_16) and 2.3 us (decode_32) in the stage-2 profile, 2.0% and 1.0% of the shape. The profile shows no measurable inter-kernel gap (108.9 + 2.2 = 111.1 vs median 110.9), so the realistic ceiling is ~2 us per shape, minus whatever the in-kernel q build adds to each CTA's critical path. Expected directions match TASK.md: weight 16 full decode_16/32 faster, everything else gated out. Floors are 63.2/126.4 us, so this stage moves the milestones by about 2%; M3 (<=1.5x floor) is unreachable by this change and is not claimed. WEAKEST LINK: the q build latency. In stage 2 the FP8 q rides the cp.async ring (issue_tile issues it with the segment's first tile, so it costs nothing). In v3 each CTA must load 2 KB of bf16 qkv, RMSNorm it and cast it to FP8 before its first QK MMA, once per segment (~17 CTAs per request at B=16, ~8 at B=32). If that compute is serialized ahead of the ring prologue it adds ~0.5-1 us to every CTA and eats most of the 2 us. sliding-fused-hoist-first-tile (regression) already showed that prologue ordering is sensitive in this codebase. Build it so the 2 KB qkv slice is cp.async'd into smem with the segment's first tile, exactly where q is fetched today, and normalized when that stage is consumed. SPEC ERROR: the change paragraph says RoPE at positions[b] with inv_freq for both q and the new k. Full layers have no RoPE (TASK.md 'rope(q,p), rope(k,p)  # sliding layers'; submission rope_theta 0 = no RoPE; master full-layer pre-attn passes ROPE=bool(theta)=False and a dummy inv_freq). Mirror master: RMSNorm (scaleless, eps 1e-5) plus the FP8 cast, no rotation. The LESSONS sin/cos caveat is therefore moot here, and k/v rows should match master almost byte for byte (only rsqrt rounding can differ). The ownership argument holds: reference.py builds cache_loc = page_table[b, pos] with cache_seqlens = pos + 1, so key seq_b - 1 is exactly cache_loc[b], and stream-K tile ranges are disjoint, so exactly one CTA owns it.

### Confounders on the current kernel
- full-decode-splitkv-s2-streamk-ticket-inproc (the depends_on parent, not on master): the measured baseline for this stage is the stage-2 profile (110.9 / 226.8 us), not BEST.json. A headline compared against best (1.3949) mixes in stage 2's own ~12/15 us gain, so read top_kernels against stage 2.
- fused-preattention-triton: its _pre_attention_kernel is the reference math being folded in. If v3's q/k differ (e.g. RoPE wrongly applied), the in-process self-check falls back to v2 and the run silently re-measures stage 2. Grep for PATH CHOSEN before interpreting anything.
- paired-heads-launch-cleanup / ab-tune-batch-gate: the gated path takes the UNPAIRED q layout (paired = ... and not splitkv). v3 must build q unpaired [16,64]; no interaction otherwise.
- full-decode-cuda-gather-splitkv (v1) stays compiled as a second-level fallback; no timing interaction.
- sliding-* landed hypotheses: gated out; sliding-shape movement is noise (stage 2 moved them by <=1 us).

### Interference and A/B needs
No okc companion. The only ambiguity is whether a ~2 us shape delta is the launch removal or noise. Shape err_ns is ~50 ns, so a 2 us move is readable, and the cleanest control is in-process: extend the stage-2 eager diagnostic (same GPU, same synthetic B=16/32 x 80000 inputs) to print (pre-attn + v2) vs v3, plus v3's kernel time alone vs v2's kernel time alone. That shows directly how much the in-kernel q/kv build costs inside the attention kernel versus the launch it saves. Remember that prepare timings undercount the judge by 20-25% (LESSONS), so rank variants with them but do not quote them as the milestone. Reverting a landed hypothesis would isolate nothing, because no landed change shares this prologue.

### Open questions
- The bigger tail on this kernel is the ticket merge (~6 us per shape per the stage-2 diagnostics), not pre-attn (2 us). Should a sibling stage target the merge, e.g. the two-level ticket from full-decode1-streamk-s3-two-level-ticket applied at B=16 (nseg ~17), or a parallel merge across the last arriver's threads?
- This branch and full-decode1-streamk-s3-two-level-ticket both edit splitkv_attn_sk_kernel from the same parent. If both pass, stage 4 has to merge v3 (fused pre-attn, 1<B<=64) with the batch-1 two-level path. Fused pre-attn is proportionally worth more at batch 1 (1.4 us of a ~14 us target shape).
- Does the q build add per-segment latency measurable in v3's own kernel time vs v2's? The eager diagnostic should print it.
- positions and inv_freq are not needed on full layers. Drop them from the v3 signature, or confirm they are unused.


## Implementation notes

Edited only `muse_attention_decode/submission.py` in the worktree (+643/-10 lines, nothing committed). Built on stage 2 as-is (relaxed arch gate `binaryVersion == 90`, effective grid `gx = min(gridDim.x, T)`, tickets re-zeroed in the `finally` after diagnostics; all kept).

- New module `muse_full_decode_splitkv_v3` (`_SK3_CPP`/`_SK3_CUDA`, built by `_build_sk(name, cpp, cuda, funcs)`; TORCH_CUDA_ARCH_LIST=9.0a around load_inline and restored; `-O3 -lineinfo`). Its source is stage 2's helper prefix (sliced from `_SK_CUDA` at import, so v2 is byte-identical) plus `splitkv_attn_sk_fused_kernel`: 4 stages, ticket protocol, same tile math and finalize as stage 2. v2 stays compiled as the fallback.
- Q build (review: overlap with the ring): the segment's first tile cp.asyncs the request's 2 KB bf16 q slice of qkv into a per-stage slot with the K/V rows. At consumption (tile landed, next 3 tiles in flight) each thread normalizes its own row (scaleless RMSNorm, eps 1e-5, 2-step shuffle across the 4 lanes of a quad), rounds to e4m3 (`__nv_cvt_float_to_fp8` SATFINITE, RN) and builds the MMA A fragments. Nothing runs ahead of the ring prologue. Unpaired q layout.
- New k/v row: the CTA whose stream-K range holds a request's last tile (key seq_b - 1 = cache_loc[b]) stores k = norm(k)/k_scale and v = v/v_scale as e4m3 while block 0's page indices are in flight (warps 0-3), `__threadfence()`, and the barrier after the index wait orders it before every ring cp.async. Tile ranges are disjoint, so no other CTA reads that row.
- Review correction applied (review wins over spec): NO RoPE. Verified: reference.py `rope_theta = ROPE_THETA if layer == "sliding" else 0.0`, TASK.md "no RoPE" on full layers, master's `_pre_attention_kernel` is launched with `ROPE=bool(theta)` = False. So positions/inv_freq are dropped from the v3 signature (answers the review's open question), and kernel_fn takes v3 only when `not theta` (else v2/master path). The spec's sin/cos tolerance caveat is moot; new-row check is still rel L2 < 5e-3.
- Dispatch in kernel_fn: `splitkv and _EXT3 is not None and not theta` -> `_full_decode_sk3` (one launch, no q tensor, no `_pre_attention_kernel`). Everything else unchanged byte for byte.
- Self-check (`_sk3_validate`, in-process, only after stage 2 validated): image check (cap 9.0, binaryVersion 90, CTAs/SM), canary B=2, 8 ragged cases (B=2/16/32/64, kv_scale 1 and 0.25, grids 3/7/device). Per case: host ownership check (cache_loc == page_table[b, seq-1]; exactly one CTA owns each last tile), master `_pre_attention_kernel` + v2 on cloned caches as reference, v3 on the originals; asserts v3 out vs FP32 ref < 2e-2 and vs v2 out < 5e-3, new k/v rows vs master rel L2 < 5e-3 and finite, all other cache bytes equal. Each launch printed before, synced and printed after. Then graph capture + 2 NaN-poisoned replays through kernel_fn, and a ticket-sum == 0 check.
- PATH CHOSEN lines: success `full decode streamk fused-preattn v3`; any v3 failure `full decode streamk v2 (fallback: <reason>)` (v2 keeps running); v2 failures keep the stage-2 messages.
- Added `_sk3_diagnostics` (eager, cold L2, B=16/32 x 80000): pre-attn + v2 vs fused v3 and v3 - v2 kernel time, as the review asked.
- Not verified: no nvcc/GPU on this node; the CUDA was only read-checked and py_compiled.

## Rules check

- [x] CUDA-graph capturable: kernel_fn v3 path only allocates workspaces from shapes and launches one kernel; no sync/.item()/value-dependent control flow (syncs are in prepare-time self-check only; replay check captures kernel_fn).
- [x] Nothing derived from `data` is cached across calls: tickets are device-allocated, zeroed once, and every launch resets them; the cache rows are outputs, not cached inputs.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] KV cache FP8 e4m3 (k/v rows stored e4m3), queries rounded to FP8 E4M3 (not lower), activations bf16.
- [x] Attention sees every key: stream-K covers all tiles of every request including the new token; no skipping.
- [x] No timing patches, harness I/O, module edits, reference import, subprocesses or network (grep of diff: no subprocess/fork/multiprocessing; load_inline is in-process).
- [x] All work inside the timed call; no output caching or replay (self-check runs at import on synthetic data only).
- [x] No triton.autotune; first-line marker `# okc: problem=muse_attention_decode` kept.

okc_lint output (`python3 -I okc_lint.py submission.py --problem muse_attention_decode`): `OK with warnings: submission.py`; 16 `warn host-sync` lines, all `torch.cuda.synchronize()` / `.item()` in prepare-time self-check code, "outside kernel_fn's call graph; fine in prepare()". No errors. `py_compile` passed; `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 (model opus), by reading only (no nvcc or GPU on the audit node). **Verdict: dispatch-with-fixes.**

Checked and found sound:
- Pre-attention parity with master `_pre_attention_kernel` (ROPE=False). Same q/k head offsets in qkv and the same scaleless RMSNorm (eps inside the sqrt). Values go fp32 to e4m3 with RN satfinite and no bf16 step, which matches Triton's `.to(float8e4nv)`. k is stored as `(x*r)/k_scale` and v as `v/v_scale`, and q carries no scale, because 5.473 and k_scale are folded into `scale` as in v2. The row address is `(cache_loc*HKV+kh)*D`. Exactly one CTA stores each request's row, the one that owns its last tile. reference.py has `cache_loc = page_table[b, seq_b-1]`, so that CTA is the only reader of the row. Its stores, then `__threadfence`, then `__syncthreads` all come before its first ring cp.async.cg, so the new key is attended.
- The q slot follows v2's per-stage slot discipline. The slot is written by `issue_tile(n+3)` after the top-of-loop barrier of iteration n, and it was last read at iteration n-1. The q slice is cp.async'd in the tile's own group and read only after `cp_wait<STAGES-2>` plus a barrier. The 16-byte alignment holds (row stride 4608 B, data_ptr is checked).
- The ticket protocol, slot indexing, `gx = min(grid, T)` and the re-zero in `finally` are byte-identical to stage 2. A v3 failure still re-zeroes the tickets before the v2 timed path runs.
- Shared memory is 99,360 B per CTA, so 2 CTAs fit per SM.
- kernel_fn's v3 path makes only shape-derived `torch.empty` calls and one launch. There is no sync, no value branching, no child process and nothing cached from `data`. Every v3 failure falls back to v2 and prints `PATH CHOSEN`.

Fixes applied:
1. The q build in `splitkv_attn_sk_fused_kernel` now makes two passes over the smem q slot (sum of squares, then scale and cast). Before, it kept `xv[4][4]` live in registers. Stage 2 already uses 121 of the 128 registers allowed at `__launch_bounds__(256, 2)`, and 16 more live floats next to acc/qa risked spills. The math is unchanged.
2. The self-check tolerance on the new k/v rows is now rel L2 < 2e-2, up from 5e-3. Triton's approximate sqrt and division versus CUDA's IEEE ones can flip a single e4m3 ulp, which is about 8e-3 of a B=2 row set. That would silently drop the run to v2. A real indexing or scale bug shows up as O(1).

Residual risk: the CUDA has never been compiled. The build adds about 40 s to import, and spills are only logged (`local` in the `[sk3]` attrs line), never gated. Check `PATH CHOSEN` and the `[sk3]` regs/local line before reading any timing.

## Result

- job `825baa03798f431a` (check), status done, passed True, tests 16/16
- speedup 1.4283x, 95% CI [1.4069, 1.4497], score 2636.9450599469105 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s3-fused-preattn.patch`; branch `hyp/muse_attention_decode/full-decode-splitkv-s3-fused-preattn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.09 | 13.49 | 1.0085 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.19 | 27.59 | 0.9855 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.19 | 15.67 | 0.9956 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 110.91 | 109.55 | 1.0175 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.29 | 24.4 | 1.0163 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.81 | 223.11 | 1.0243 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 16.83 | 17.35 | 0.9922 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 29.84 | 30.04 | 0.9978 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.72 | 23.23 | 0.9953 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.42 | 163.39 | 1.0136 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.87 | 29.6 | 0.9962 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.06 | 300.35 | 1.0143 | flat |

no shape moved outside noise. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 3 of `_pre_attention_kernel launch on full decode 1<B<=64 (folded into splitkv_attn_sk_kernel)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode-splitkv-s3-fused-preattn --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-splitkv-s3-fused-preattn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

