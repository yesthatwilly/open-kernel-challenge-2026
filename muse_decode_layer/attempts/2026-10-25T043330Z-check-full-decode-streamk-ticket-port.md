# muse_decode_layer · 2026-10-25T043330Z · check · full-decode-streamk-ticket-port

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T042121Z-full-decode-streamk-ticket-port.md` (branch `hyp/muse_decode_layer/full-decode-streamk-ticket-port`).

- **Job:** `6d48932b230d4067` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `c7fca44`

## Goal
Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed restack). Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_decode's stream-K kernel from branch hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc (splitkv_attn_sk_kernel: stream-K over 64-token tiles of the concatenated page-1 KV of all requests, grid 2 x SM count, each CTA owns an equal contiguous tile range; per-request ticket: each CTA writes its fp32 partial (acc, m, l), __threadfence(), atomicAdd on ticket[b]; the last arrival LSE-merges, applies v_scale and the RMSNorm x sigmoid(gate) epilogue, stores bf16 and resets ticket[b] to 0 for graph replay). In decode_layer replace the two calls in _splitkv_attn/_splitkv_combine (kernel A + kernel B) with the one stream-K launch at the same gate (_splitkv_gate: full layer, q_len 1, page 1, 1 < batch <= 64); it consumes the same q, KV cache, page_table, cache_seqlens, v_scale and qkv gate columns master's combine reads, and writes the same bf16 out tensor the o GEMM reads. The ticket buffer is a module-level int32 zero buffer allocated at import; the partial workspace is sized from config (max batch x grid) in prepare. Keep master's kernel A/B compiled as the fallback. Self-check vs master's kernels and an fp32 reference at B in {2,5,16,32,33}, agentic and uniform lengths, kv_scale 1 and 0.25, plus two consecutive graph replays (ticket reset). BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (sm_90a only) and pass -std=c++20 if extra_cuda_cflags are overridden; print the build line, regs/smem/local per kernel and binaryVersion; the first launch is a tiny synthetic canary followed by torch.cuda.synchronize() and a 'launching/ok' print naming the kernel; then the self-check, all IN-PROCESS (never a child process: the rules reject starting processes). Any failure keeps the incumbent for the gated shapes and prints 'PATH CHOSEN: ... (fallback: <reason>)'. The prepare-time selector must print the CUDA-event timing of new vs incumbent at the gated batch and keep the incumbent if the new kernel is slower (LESSONS 03:41Z: never ship a kernel its own timing says is slower). No timing, sync or allocation in kernel_fn; gates read config (batch, q_len, window, page) only. First measurement is an okc check.

## What changed, and where
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

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path only launches the kernel and allocates the bf16 out by shape (as master); workspace and tickets are preallocated; all synchronize/.item() calls are in prepare (lint: warnings only, all "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls: the ticket, workspace and dummies are shape/device-only scratch; the selector uses synthetic data only.
- [x] Weights untouched (n/a: no weight change).
- [x] Activations BF16, KV cache FP8, query FP8 E4M3 (same q_fp8 as the incumbent); softmax/accumulate fp32.
- [x] Attention sees every key: stream-K covers every 64-token tile of every request; no skipping.
- [x] No timing patches, harness frames, subprocesses or network (grepped the diff for subprocess/multiprocessing/fork/Popen: none; `os` is used only for the TORCH_CUDA_ARCH_LIST env around load_inline).
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: muse_decode_layer/submission.py` (only prepare-only host-sync warnings). py_compile OK; __pycache__ removed.

Biggest risk: first judge compile and launch of the module (nvcc with `-std=c++20` plus sm_90a-only arch list, 2 CTAs/SM register/smem fit unverified here) and a possible selector that keeps A+B on noise. No GPU on this node, nothing was run.

## Risks / what could fail
medium
Kernel proven on the judge in attention (no fault, sm_90a). Decode-layer risk is interface only (gate columns/out layout). Compare against master's split-KV rows (c243a29f).

## Result (after the run)
- job `6d48932b230d4067` (check), status done, passed True, tests 13/13
- speedup 1.2775x, 95% CI [1.2583, 1.2966], score 17707.133145610635 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2855726646854595, "ci": [1.2662890747151776, 1.3048562546557412], "job": "7a9f5de5aecc4aba"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-streamk-ticket-port.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-streamk-ticket-port.patch`; branch `hyp/muse_decode_layer/full-decode-streamk-ticket-port`
