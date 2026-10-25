---
problem: muse_decode_layer
slug: full-decode-streamk-ticket-port
created: 2026-10-25T042121Z
base: master
area: "FA3 call (full decode, batch > 1): splitkv_attn_kernel + splitkv_combine_kernel"
change: >
  Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed
  restack). Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA
  split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's stream-K
  kernel from branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc
  (splitkv_attn_sk_kernel: stream-K over 64-token tiles of the concatenated page-1 KV of all
  requests, grid 2 x SM count, each CTA owns an equal contiguous tile range; per-request ticket:
  each CTA writes its fp32 partial (acc, m, l), __threadfence(), atomicAdd on ticket[b]; the last
  arrival LSE-merges, applies v_scale and the RMSNorm x sigmoid(gate) epilogue, stores bf16 and
  resets ticket[b] to 0 for graph replay). In decode_layer replace the two calls in
  _splitkv_attn/_splitkv_combine (kernel A + kernel B) with the one stream-K launch at the same
  gate (_splitkv_gate: full layer, q_len 1, page 1, 1 < batch <= 64); it consumes the same q, KV
  cache, page_table, cache_seqlens, v_scale and qkv gate columns master's combine reads, and
  writes the same bf16 out tensor the o GEMM reads. The ticket buffer is a module-level int32 zero
  buffer allocated at import; the partial workspace is sized from config (max batch x grid) in
  prepare. Keep master's kernel A/B compiled as the fallback. Self-check vs master's kernels and
  an fp32 reference at B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, plus
  two consecutive graph replays (ticket reset). BUILD/SAFETY (LESSONS 2026-10-25):
  TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden;
  print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny
  synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the
  kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting
  processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ...
  (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs
  incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z:
  never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn;
  gates read config (batch, q_len, window, page) only. First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced: splitkv_attn 112.7 + combine 8.4 = 121.1 us (profile c243a29f); attention measured -9.3% kernel A from stream-K and -3..5% from the in-kernel ticket combine -> ~106-108 us (KV floor ~63). Shape 306.7 -> ~294. Kill: replaced-kernel time >= 116 us at B=16 in the profile"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced 248.5 + 8.6 = 257.1 us; stream-K removes the agentic tail imbalance at B=32 -> ~226-230 (floor ~126). Shape 479.6 -> ~452. Kill: >= 245 us"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: outside the gate
  - shapes: "*layer=sliding*"
    direction: flat
    why: outside the gate
  - shapes: "*load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64
review: proceed
status: done
review_at: 2026-10-25T042453Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T042453Z
base_sha: f6e653b6cc358b21c20086af784b37d069a021a5
base_blob: c9067b8201f2480c64910c4a71098df7ce616f73
branch: hyp/muse_decode_layer/full-decode-streamk-ticket-port
worktree: .worktrees/muse_decode_layer/full-decode-streamk-ticket-port
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043330Z
impl_commit: c7fca44d4e87242a87fcd253b40eae128912e9f2
submission_sha256: a69a00ee5d71779ad51f9fb05824d398a416f7d62f2609e73d64723e081d4ed9
norm_sha256: 07b790c1ca28561819a451e3ec1ad0a5a4f52bf75bd1ae58b75d60d378d0f1a7
submission_blob: 14a7c7be22acb26d22510ac7c55ea96451f9c483
lint_warnings: 44
attempt_file: 2026-10-25T043330Z-check-full-decode-streamk-ticket-port.md
prepared_at: 2026-10-25T043330Z
job: 6d48932b230d4067
dispatched_at: 2026-10-25T043340Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.277468227859758
speedup_ci: [1.2583062044418616, 1.2966302512776542]
harvested_at: 2026-10-25T053800Z
best_at_harvest: 1.2855726646854595
promotion: null
---

# full-decode-streamk-ticket-port

## Change

Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed restack). Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's stream-K kernel from branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles of the concatenated page-1 KV of all requests, grid 2 x SM count, each CTA owns an equal contiguous tile range; per-request ticket: each CTA writes its fp32 partial (acc, m, l), __threadfence(), atomicAdd on ticket[b]; the last arrival LSE-merges, applies v_scale and the RMSNorm x sigmoid(gate) epilogue, stores bf16 and resets ticket[b] to 0 for graph replay). In decode_layer replace the two calls in _splitkv_attn/_splitkv_combine (kernel A + kernel B) with the one stream-K launch at the same gate (_splitkv_gate: full layer, q_len 1, page 1, 1 < batch <= 64); it consumes the same q, KV cache, page_table, cache_seqlens, v_scale and qkv gate columns master's combine reads, and writes the same bf16 out tensor the o GEMM reads. The ticket buffer is a module-level int32 zero buffer allocated at import; the partial workspace is sized from config (max batch x grid) in prepare. Keep master's kernel A/B compiled as the fallback. Self-check vs master's kernels and an fp32 reference at B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, plus two consecutive graph replays (ticket reset). BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## Expected

- `*layer=full load=decode_16*` faster: replaced: splitkv_attn 112.7 + combine 8.4 = 121.1 us (profile c243a29f); attention measured -9.3% kernel A from stream-K and -3..5% from the in-kernel ticket combine -> ~106-108 us (KV floor ~63). Shape 306.7 -> ~294. Kill: replaced-kernel time >= 116 us at B=16 in the profile
- `*layer=full load=decode_32*` faster: replaced 248.5 + 8.6 = 257.1 us; stream-K removes the agentic tail imbalance at B=32 -> ~226-230 (floor ~126). Shape 479.6 -> ~452. Kill: >= 245 us
- `*load=decode_1 *` flat: outside the gate
- `*layer=sliding*` flat: outside the gate
- `*load=verify_*` flat: outside the gate

## Risk

Kernel proven on the judge in attention (no fault, sm_90a). Decode-layer risk is interface only (gate columns/out layout). Compare against master's split-KV rows (c243a29f).

## Milestones

Rewrite stage 3 of `splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042453Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, already-proven kernel port that replaces exactly the two incumbent split-KV kernels (A 112.7 + B 8.4 us at decode_16, 248.5 + 8.6 at decode_32 in c243a29f) at an unchanged config gate. Judge it on the replaced-kernel rows: the headline gain (~0.5-1%) is below the CI width. Port the kernel from attention master c3c1b01 (s2 path = template <4,true,false>) rather than the s2 branch, so the sibling full-decode1-two-level-ticket-port becomes a disjoint batch==1 gate on the same module.

### Justification
A. Mechanism: the hypothesis's replaced work is kernel A (persistent 2 CTA/SM, strided 1024-token chunks) plus kernel B (separate LSE combine + norm-gate). Stream-K over 64-token tiles gives each CTA an equal, contiguous range, which removes the agentic-length tail and the strided page-table walk. The per-request ticket folds B into the last-arriving CTA and drops one graph node. On the same H200 judge, muse_attention_decode measured the same kernel as follows. decode_16: 119.3 us (A+B) -> 108.9 us (s2 e592b1f5) and 110.1 us (s3 2a324178). decode_32: 254.1 -> 226.8 and 225.6. Both runs passed 16/16 tests with no fault on sm_90a. Decode_layer's incumbent rows are nearly identical (A 112.7 vs 111.4; A 248.5 vs 245.5), so the 106-108 and 226-230 us targets transfer directly. The KV floor (~63/126 us) is far above L2, so the surrounding layer's L2 state should not change the kernel's bandwidth. At T=16/32 the dense chain is active and the side stream is unused, so the kernel runs alone on the GPU with the same occupancy assumptions as in attention. Expected shapes and directions are consistent. The gate is unchanged (full, q_len 1, page 1, 1<B<=64, tp8), so decode_1, sliding and verify_* stay flat. Weight 16 per full decode shape means only about -13 us at decode_16 (306.7 -> ~294, ~4%) and about -27 us at decode_32 (479.6 -> ~452, ~6%). That is roughly 0.5-1% on the headline, inside the +/-1.5% CI. Marlin rows on these shapes swing 10-30 us run to run (kv-bw-3p5: Marlin 95.9 -> 106.9 at decode_16, 111.2 -> 144.7 at decode_32), so even the per-shape totals can hide a 9-12 us kernel gain. Read splitkv_attn_sk_kernel's top_kernels row against 121.1/257.1 (kill >= 116/245, as written). WEAKEST LINK: the prepare-time selector. prepare(weights) has no config, so 'timing at the gated batch' must use synthetic B=16/32. With uniform lengths, stream-K's advantage (the ragged tail) is smallest and tickets vs kernel B is a ~8 us difference, so a noisy CUDA-event comparison could silently keep A/B. The run would then read as 'flat' while measuring the incumbent. Use agentic-style ragged synthetic lengths at 80000 ctx, choose per batch (16 and 32 separately) with a small margin, and print PATH CHOSEN per batch so the profile row can be read as which kernel actually ran. A secondary weak point: prepare cannot size the workspace 'from config', so use a fixed MAXB=64 x (grid + B) allocation (~1.3 MB, as in the s3 code's slots <= grid + B).

### Confounders on the current kernel
- full-decode-splitkv-restack-on-pdl: the incumbent being replaced, not a confounder. The port must keep its q prep (q.view(tokens,16,64).to(FP8).contiguous()), read gate through qkv.stride(0) like splitkv_combine, and keep v_scale and RMS_EPS exactly as kernel B does. A mismatch there shows up as a test failure, not a timing confound.
- small-t-launch-minimal-dense-chain-pdl: active at T=16/32 (DL_MAX_TOKENS=32) but downstream of the o GEMM. The attention kernels are plain <<<>>> launches with no PDL edge, so the only interaction is the one fewer graph node (~1-3 us), which counts as part of this change. It also keeps the side stream off at these shapes, so no concurrent kernels skew stream-K's static partition.
- fa3-hdim128-head-pairing-batched: inactive on the gated shapes (paired requires 'not splitkv'), so no confound.
- stack-align-par-sidestream-sliding1: its side-stream dispatch runs only at T>32 (verify_16/32), outside the gate, so no concurrency with the attention kernel.
- Sibling full-decode1-two-level-ticket-port (queued alongside, not landed): same kernel family, disjoint gate (batch == 1 vs 1 < batch <= 64), so each run attributes cleanly. Attention master c3c1b01 holds both paths in one template: TWO=false is this port's B>1 single-level ticket at grid 2xSM; TWO=true is the sibling's B=1 two-level ticket at grid = SM count. In attention, adding TWO left B>1 flat (108.9/226.8 -> 110.1/225.6 us). Both ports built separately from master will conflict when restacked (same module and gate code), and the second lander must restack. If both take the kernel source from c3c1b01 under one module name, the second becomes a gate/grid edit only. The sibling's text 'batch>1 keeps master kernel A/B' goes stale if this one lands first.

### Interference and A/B needs
No companion is warranted. Kind is rewrite stage 3, judged on the replaced kernels' own rows (splitkv_attn_sk_kernel vs splitkv_attn_kernel + splitkv_combine_kernel), and those rows separate cleanly. The gate is config-only and unchanged, and nothing landed overlaps the gated shapes (pairing is off, PDL is not on the attention edge, the side stream is idle at T<=32). An ab-revert of the incumbent restack would just compare against paired FA3, which is already known. A control isolating stream-K from ticket-fold would only re-measure attention's s2 diagnostics (stream-K with and without ticket), so ask for those in-process prints instead of a separate check. The one ambiguity is which path the prepare selector actually picked. That is a logging requirement (per-batch PATH CHOSEN plus selector timings), not a companion run.

### Open questions
- Will the implementer port from attention master c3c1b01 (s2 kernel plus the unused TWO template path) or from the s2 branch f291b49? c3c1b01 is preferred so the batch-1 sibling reuses the same module.
- Selector inputs: which synthetic length distribution and batch values? Uniform 80000 understates stream-K's benefit. Does it choose per batch (16 vs 32) or once for all gated batches?
- Workspace sizing without config in prepare: fixed MAXB=64 x (2*SMs + 64) slots, or allocated per call as attention s2 did? The hypothesis says 'no allocation in kernel_fn', but master already allocates there.
- The decode_layer profile is dist=soft while attention is dist=coherent. That is the MoE routing distribution and does not affect attention, but confirm the agentic KV-length spread is the same generator in both problems.


## Implementation notes

All edits are in `muse_decode_layer/submission.py` of the worktree (+918/-2).

What changed:
- New module `muse_decode_sk_ticket` (load_inline at import, `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored; arch flags printed; `-O3 -std=c++20 -lineinfo`). The CUDA source is attention master c3c1b01's stream-K block (lines 942-1407 of its submission.py) copied verbatim: `splitkv_attn_sk_kernel<STAGES, TICKET, TWO>`, merge/finalize helpers, two-level code included. Only `<4, true, false>` is instantiated (`sk_pick`), `sk_attn` rejects any variant but 0. The attention-only diagnostic kernels (`splitkv_sk_combine_kernel`, no-ticket and 5-stage variants, `sk_combine`) were dropped. The sibling full-decode1-two-level-ticket-port can add `<4, true, true>` plus its batch == 1 gate under the same module name.
- `_sk_launch` / `_full_decode_sk`: one launch replacing `_splitkv_attn` + `_splitkv_combine`; same q (FP8 [B,16,64]), KV cache, page_table, cache_seqlens, k/v_scale and qkv gate columns (`qkv.stride(0)`), writes the same `[B,1,Hq*D]` bf16. Kernel A/B and its module stay compiled as the fallback.
- Gate `_sk_gate(cfg)` = `_SK_OK and _splitkv_gate(cfg) and _SK_CHOICE[16 if batch <= 23 else 32]`, config-only. In kernel_fn the splitkv branch calls the stream-K launch when it is true, else kernels A+B.
- Ticket buffer: module-level int32 [64] zeros allocated at import; zeroed in prepare outside graph capture before the self-check and again after it (and after each selector run and at the end). The sum after the self-check is printed and must be 0.
- Workspace: persistent fixed buffers (2 x SM + 64 slots of [16,64] fp32 and [2,16] fp32) allocated in prepare, so kernel_fn does not allocate them (it still allocates the bf16 out per call, as master does).
- `_sk_prepare` (called from prepare after `_splitkv_prepare`): capability (9,0), `sk_attrs` print (binaryVersion, regs, local, smem, CTAs/SM) and cubin==90 check, canary B=2 [4,66] with launching/returned prints and sync, then 13 cases with a flushed print before and a synchronize + print after every launch: B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, grids device/3/7/40, a B=16 batch with seq%64 in 1..7 and sequences < 64 (1, 7, 33, 63, 64, 65), all vs the fp32 reference (rel L2 < 2e-2, finite). Two CUDA-graph replays (twice each, NaN-poisoned output) on the tails batch and on an agentic B=16 kv 0.25 batch check the ticket reset.
- Selector `_sk_select(b)` for b in 16 and 32: synthetic agentic lengths (uniform 80000 +/- 7.5%, ctx 80000, as reference.py's spread), checks stream-K vs the incumbent A+B output (rel L2 < 1e-2), then times both interleaved with CUDA events (7 reps x 10 calls, median), picks stream-K only if it is faster by 1% (ratio < 0.99), and prints `PATH CHOSEN: full decode attention batch B: ...` with both timings. Any failure prints `PATH CHOSEN ... (fallback: <reason>)` and keeps A+B; nothing raises.

Deviations / review fixes applied (the review wins over the spec):
- Kernel source from attention c3c1b01 and module name `muse_decode_sk_ticket`, TWO=false only (review).
- Selector at agentic ragged lengths per batch (16 and 32 separately) with PATH CHOSEN per batch, fixed MAXB=64-style workspace instead of "sized from config" (review). Gated batches 2..23 follow the B=16 decision, 24..64 follow B=32 (my choice; the judged shapes are 16 and 32).
- Tickets zeroed outside capture and after the self-check; partials merged in slot order (merge_slots loops c_lo..c_hi) (review).
- Self-check covers seq%64 in 1..7 and sequences < 64 (review).
- sm_90a: the kernel uses only mma.sync, cp.async and cvt e4m3x2 (no wgmma/setmaxnreg), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard was added; the 9.0a build and cubin check are kept as in attention.
- No spin waits exist in this kernel (no CTA waits on another), so no bounded wait was added; faults are caught by the per-launch prints.
- Judge on the `splitkv_attn_sk_kernel` row of the profile (kill if >= 116 us at decode_16 / 245 us at decode_32), not the headline; PATH CHOSEN tells which kernel ran. If the selector keeps A+B the run measures the incumbent.

## Rules check

- [x] CUDA-graph capturable: kernel_fn path only launches the kernel and allocates the bf16 out by shape (as master); workspace and tickets are preallocated; all synchronize/.item() calls are in prepare (lint: warnings only, all "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls: the ticket, workspace and dummies are shape/device-only scratch; the selector uses synthetic data only.
- [x] Weights untouched (n/a: no weight change).
- [x] Activations BF16, KV cache FP8, query FP8 E4M3 (same q_fp8 as the incumbent); softmax/accumulate fp32.
- [x] Attention sees every key: stream-K covers every 64-token tile of every request; no skipping.
- [x] No timing patches, harness frames, subprocesses or network (grepped the diff for subprocess/multiprocessing/fork/Popen: none; `os` is used only for the TORCH_CUDA_ARCH_LIST env around load_inline).
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: muse_decode_layer/submission.py` (only prepare-only host-sync warnings). py_compile OK; __pycache__ removed.

Biggest risk: first judge compile and launch of the module (nvcc with `-std=c++20` plus sm_90a-only arch list, 2 CTAs/SM register/smem fit unverified here) and a possible selector that keeps A+B on noise. No GPU on this node, nothing was run.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only emulation, no GPU). **Verdict: dispatch** (no fixes).

- Kernel source: diffed against attention c3c1b01 lines 942-1508. The kernel body and its helpers are byte-identical. Only the diagnostics (`splitkv_sk_combine_kernel`, `sk_combine`) and the variant 1/2/3 dispatch were removed, and `sk_pick` is pinned to `<4, true, false>`.
- Glue matches the incumbent. q is the same `q.view(T,16,64).to(FP8).contiguous()`. Cache layout, contiguity and `page_table.stride(1)==1` checks are the same as kernel A, and `pt_stride = page_table.stride(0)`. The gate is read at `(HQ+2*HKV)*D` via `qkv.stride(0)`, as in kernel B. `v_scale` and `RMS_EPS` are applied as in kernel B. The output is the same `[B,1,Hq*D]` bf16.
- Workspace is sized 2·SM+64 slots, which covers the maximum slot bx+b < grid+B. The slot pairs (bx,b) cannot collide because CTA and request tile ranges are both monotone.
- Gate is config-only. `_SK_CHOICE` is fixed in prepare. Every gated batch in TASK (16, 32, and the B=16/B=33 tests) is self-checked for correctness. For B=33 that check is the B=33 ragged and uniform cases.
- Tickets are zeroed in prepare outside capture and checked after the self-check and after each selector run. The last arriver resets its ticket, so the counters are back to 0 by the next graph replay. The replay test covers this, with NaN-poisoned output.
- Build: `-std=c++20` matches torch's own default. The bf16 cxx20 review showed this torch requires C++20. `TORCH_CUDA_ARCH_LIST=9.0a` is set around load_inline and restored in `finally`. The binaryVersion==90 check and the 90a-only flag check can pass.
- Selector equivalence gate: in a numpy emulation at ctx 80000 (fp16 P, chunk-1024 vs ragged stream-K segments), the worst rel L2 between stream-K and A+B was 3e-4, against the 1e-2 gate. It does not always fail.
- Selector timing: eager, CUDA events, interleaved, median of 7×10, with no explicit L2 flush and no graph replay. The KV stream is 327/655 MB per call, so L2 is effectively cold. The eager A→B launch gap biases the comparison by a few µs toward stream-K. That is smaller than the expected ~10 µs gain, and the 1% margin (~1.2 µs) cannot flip a ~9% win.
- kernel_fn has no prints, try/except, syncs or per-call state. The only per-call allocation is the bf16 out, as on master. okc_lint: OK with warnings (all prepare-only). py_compile OK.

Residual risk: the first judge compile and launch of the new module (2 CTAs/SM fit unverified), and a ~2% eager-timing bias in the selector that only matters if the true gain is under ~2%.

## Result

- job `6d48932b230d4067` (check), status done, passed True, tests 13/13
- speedup 1.2775x, 95% CI [1.2583, 1.2966], score 17707.133145610635 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2855726646854595, "ci": [1.2662890747151776, 1.3048562546557412], "job": "7a9f5de5aecc4aba"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-streamk-ticket-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-streamk-ticket-port.patch`; branch `hyp/muse_decode_layer/full-decode-streamk-ticket-port`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 126.15 | 126.54 | 0.9838 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.7 | 143.8 | 0.9859 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 209.82 | 204.88 | 0.9988 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 306.72 | 296.35 | 1.0358 | gained |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 242.33 | 237.43 | 1.0009 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 479.6 | 437.49 | 1.072 | gained |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 187.97 | 181.59 | 1.0065 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 201.51 | 204.62 | 0.9948 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 420.42 | 427.04 | 1.0199 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 564.2 | 565.43 | 0.9983 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 600.82 | 614.51 | 1.0069 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 864.67 | 878.72 | 0.9992 | flat |

gained on full decode_16, decode_32. Expected hit: 5/5. Noise ±1.5%.

## Follow-ups

- rewrite stage 3 of `splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_decode_layer "<what to fix or tune>" --kind rewrite --depends-on full-decode-streamk-ticket-port --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-streamk-ticket-port`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

