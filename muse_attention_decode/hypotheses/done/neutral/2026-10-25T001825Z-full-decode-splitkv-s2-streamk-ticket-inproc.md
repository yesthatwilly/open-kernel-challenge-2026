---
problem: muse_attention_decode
slug: full-decode-splitkv-s2-streamk-ticket-inproc
created: 2026-10-25T001825Z
base: master
area: full-layer attention (decode, splitkv kernel A/B)
change: >
  Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A +
  splitkv_combine_kernel B in load_inline module muse_full_decode_splitkv_v1; gate _splitkv_gate =
  full, q_len 1, 1 < batch <= 64, tp8, page 1 is unchanged). Per REWRITE_READ_2026-10-25 section
  1, keep the 2-CTA/SM, 256-thread, all-thread cp.async.cg 4-stage ring and the m16n8k16 f16
  compute exactly; change only the work split and the epilogue, in a NEW module
  muse_full_decode_splitkv_v2 (new kernel splitkv_attn_sk_kernel; stage-1 A/B stay compiled in v1
  as the fallback). (1) STREAM-K over 64-token tiles instead of 1024-token chunks: every CTA loads
  cache_seqlens[0..B-1] to smem, computes tiles_b = ceil(seq_b/64) and the prefix TP[b] (TP[B] =
  Ttot), and owns the flat contiguous tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math),
  which may span requests; page indices for the next 16 tiles are prefetched into double-buffered
  smem with 4-byte cp.async (zero-fill past each request end) and the ring runs across segment
  boundaries (q rows of the next request fetched into a second smem slot when a segment starts).
  The contiguous per-CTA range is also the L2/TLB-locality change for decode_32 (each CTA walks
  one contiguous page-table run instead of strided 1024-chunks). Each CTA writes one fp32 partial
  (acc[16,64], m[16], l[16]) per request segment at slot c + b (slots <= grid + B; workspace
  allocated per call from grid and batch); a segment that covers a whole request skips the
  workspace. (2) FOLD COMBINE + NORM-GATE INTO A with a per-request ticket: after writing a
  partial, __threadfence(), then one thread does atomicAdd(&ticket[b], 1); the CTA that draws
  nseg_b - 1 (nseg_b = number of CTAs whose range intersects request b, recomputed from TP) does
  __threadfence(), LSE-merges the request's slots (exact inverse c_lo..c_hi of floor(c*Ttot/grid),
  empty segments weight 0), times v_scale, then the stage-1 B epilogue math (scaleless RMSNorm
  over 64, eps 1e-5, times sigmoid(gate) from qkv, fp32) and stores bf16 to the [B,1,1024] output,
  then resets ticket[b] = 0 so graph replay starts clean. ticket is a module-level int32[64] zero
  tensor allocated once at import (not derived from inputs). Launches on full decode_16/32 drop
  from three (pre-attn, A, B) to two. (3) Decode_32 bandwidth: grid stays 2 x SM count and 4
  stages for the timed path; the import-time self-check additionally prints eager diagnostic
  timings at synthetic B=16 and B=32 x 80000 of stage-1 A+B, stream-K without ticket (+B), and
  stream-K with ticket, and of 5 stages (80 KB ring, still 2 CTA/SM) — diagnostics only, they do
  not choose the path. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (-gencode
  arch=compute_90a,code=sm_90a only), -lineinfo; run the first launch and the self-check (B in
  {2,16,32,64}, ragged lengths incl. seq % 64 in 1..7, lengths < 64, one request spanning every
  CTA, kv_scale 1 and 0.25, grids 3/7/device, two consecutive replays to prove the ticket reset)
  IN-PROCESS (never in a child process: the rules statically reject submissions that start
  processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the
  sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc
  -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for splitkv_attn_sk_kernel; anything else turns the path OFF), then a
  tiny synthetic canary launch of splitkv_attn_sk_kernel (synthetic B=2, short lengths), each
  followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the
  self-check; if the build, the image check, the canary or the self-check fails, fall back to
  master (stage-1 A+B) and print 'PATH CHOSEN: full decode splitkv v1 (fallback: <reason>)'; on
  success print 'PATH CHOSEN: full decode streamk+ticket v2'. Every other shape keeps master code
  byte for byte. The first measurement of this hypothesis must be an `okc check`, never a scored
  submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: "replaced kernels master A+B 111.4+7.9 = 119.3 us, floor 63.2. M1: 16/16 tests and <= 119.3. M2: <= 108 us (B launch gone ~-8, tail removed). M3: <= 95 us (1.5x floor)"
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "replaced kernels master A+B 245.5+8.6 = 254.1 us at 2.67 TB/s, floor 126.4. M1: <= 254.1. M2: <= 222 us (ticket -8.6 plus contiguous stream-K ranges back to ~2.9 TB/s). M3: <= 190 us"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside _splitkv_gate
risk: medium
depends_on: full-decode-cuda-gather-splitkv
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: "full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)"
review: proceed
status: done
review_at: 2026-10-25T002131Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T002131Z
base_sha: eacc593b09a9f93781b074c1c92079abb1a0756e
base_blob: c6854f093588da524e63657576b27ed720756e32
branch: hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc
worktree: .worktrees/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T003701Z
impl_commit: f291b49c49670595d0615e749294f69bc15de1a4
submission_sha256: f49bdef359197810d176338f1bf491251d9a22b13beadb48c8647f69cbe5628c
norm_sha256: 61164dd99ca40724dc1494fa08e554f6c14e08d7036277822482a5828fc343a4
submission_blob: 695c97d2e9fef8f806ec02d2fe14a02668989e3f
lint_warnings: 10
attempt_file: 2026-10-25T003701Z-check-full-decode-splitkv-s2-streamk-ticket-inproc.md
prepared_at: 2026-10-25T003701Z
job: e592b1f5d3db4f83
dispatched_at: 2026-10-25T003708Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4170903504971584
speedup_ci: [1.395833995239701, 1.4425582674953836]
harvested_at: 2026-10-25T013610Z
best_at_harvest: 1.3948738389144513
promotion: null
---

# full-decode-splitkv-s2-streamk-ticket-inproc

## Change

Stage 2 of the full-decode rewrite, built on master (stage 1 kernels splitkv_attn_kernel A + splitkv_combine_kernel B in load_inline module muse_full_decode_splitkv_v1; gate _splitkv_gate = full, q_len 1, 1 < batch <= 64, tp8, page 1 is unchanged). Per REWRITE_READ_2026-10-25 section 1, keep the 2-CTA/SM, 256-thread, all-thread cp.async.cg 4-stage ring and the m16n8k16 f16 compute exactly; change only the work split and the epilogue, in a NEW module muse_full_decode_splitkv_v2 (new kernel splitkv_attn_sk_kernel; stage-1 A/B stay compiled in v1 as the fallback). (1) STREAM-K over 64-token tiles instead of 1024-token chunks: every CTA loads cache_seqlens[0..B-1] to smem, computes tiles_b = ceil(seq_b/64) and the prefix TP[b] (TP[B] = Ttot), and owns the flat contiguous tile range [c*Ttot/grid, (c+1)*Ttot/grid) (int64 math), which may span requests; page indices for the next 16 tiles are prefetched into double-buffered smem with 4-byte cp.async (zero-fill past each request end) and the ring runs across segment boundaries (q rows of the next request fetched into a second smem slot when a segment starts). The contiguous per-CTA range is also the L2/TLB-locality change for decode_32 (each CTA walks one contiguous page-table run instead of strided 1024-chunks). Each CTA writes one fp32 partial (acc[16,64], m[16], l[16]) per request segment at slot c + b (slots <= grid + B; workspace allocated per call from grid and batch); a segment that covers a whole request skips the workspace. (2) FOLD COMBINE + NORM-GATE INTO A with a per-request ticket: after writing a partial, __threadfence(), then one thread does atomicAdd(&ticket[b], 1); the CTA that draws nseg_b - 1 (nseg_b = number of CTAs whose range intersects request b, recomputed from TP) does __threadfence(), LSE-merges the request's slots (exact inverse c_lo..c_hi of floor(c*Ttot/grid), empty segments weight 0), times v_scale, then the stage-1 B epilogue math (scaleless RMSNorm over 64, eps 1e-5, times sigmoid(gate) from qkv, fp32) and stores bf16 to the [B,1,1024] output, then resets ticket[b] = 0 so graph replay starts clean. ticket is a module-level int32[64] zero tensor allocated once at import (not derived from inputs). Launches on full decode_16/32 drop from three (pre-attn, A, B) to two. (3) Decode_32 bandwidth: grid stays 2 x SM count and 4 stages for the timed path; the import-time self-check additionally prints eager diagnostic timings at synthetic B=16 and B=32 x 80000 of stage-1 A+B, stream-K without ticket (+B), and stream-K with ticket, and of 5 stages (80 KB ring, still 2 CTA/SM) — diagnostics only, they do not choose the path. BUILD/SAFETY (LESSONS 2026-10-25): TORCH_CUDA_ARCH_LIST=9.0a (-gencode arch=compute_90a,code=sm_90a only), -lineinfo; run the first launch and the self-check (B in {2,16,32,64}, ragged lengths incl. seq % 64 in 1..7, lengths < 64, one request spanning every CTA, kv_scale 1 and 0.25, grids 3/7/device, two consecutive replays to prove the ticket reset) IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): first confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for splitkv_attn_sk_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of splitkv_attn_sk_kernel (synthetic B=2, short lengths), each followed by torch.cuda.synchronize() and a print so a fault names its kernel, then the self-check; if the build, the image check, the canary or the self-check fails, fall back to master (stage-1 A+B) and print 'PATH CHOSEN: full decode splitkv v1 (fallback: <reason>)'; on success print 'PATH CHOSEN: full decode streamk+ticket v2'. Every other shape keeps master code byte for byte.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*layer=full load=decode_16*` faster: replaced kernels master A+B 111.4+7.9 = 119.3 us, floor 63.2. M1: 16/16 tests and <= 119.3. M2: <= 108 us (B launch gone ~-8, tail removed). M3: <= 95 us (1.5x floor)
- `*layer=full load=decode_32*` faster: replaced kernels master A+B 245.5+8.6 = 254.1 us at 2.67 TB/s, floor 126.4. M1: <= 254.1. M2: <= 222 us (ticket -8.6 plus contiguous stream-K ranges back to ~2.9 TB/s). M3: <= 190 us
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside _splitkv_gate

## Risk

supersedes full-decode-splitkv-s2-streamk-ticket, whose change requires a child-process self-check

## Milestones

Rewrite stage 2 of `full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002131Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage 2 that follows REWRITE_READ section 1 exactly (keep the proven 2-CTA/SM cp.async ring, fold combine via ticket, stream-K tiles), with no rules problem; but the decode_32 'L2/TLB locality' bandwidth-recovery claim is unsupported because reference.py builds the page table from a global randperm, so the decode_32 M2 target (<=222 us) is unlikely and should be read from the in-process diagnostics, not the headline.

### Justification
Profile (FINDINGS, job 698d4cd3838c4a63): full decode_16 = splitkv_attn_kernel 111.8 + splitkv_combine_kernel 8.2 + pre-attn 2.1 us (floor 63.2); decode_32 = 245.9 + 8.4 + 2.0 us (floor 126.4). Kernel A is 92-96% of each shape, at ~2.94 and ~2.67 TB/s. The ticket fold removes one launch and the separate B kernel, but the LSE merge still runs as a serialized tail in the last-arriving CTA per request. That CTA reads about grid/B slots x (16x64+32) fp32: roughly 17 slots (~70 KB) at B=16 and 8 slots at B=32, all L2-hot. So the realistic net saving is ~4-7 us, not the full 8. splitkv-pdl-attn-to-combine was neutral, so the launch gap itself is small; the saving comes from B's own body. Stream-K over 64-token tiles removes the 1024-chunk tail, which is about 4.7 items/CTA at B=16 and about 9.5 at B=32. But splitkv-chunk2048-batch32 (doubling item size, so a worse tail) came out neutral, which says the tail is not the decode_32 limiter. WEAKEST LINK: the claim that contiguous per-CTA tile ranges restore decode_32 bandwidth through L2/TLB locality. reference.py lines 187-192 fill page_table from torch.randperm(total), so KV row addresses are uniformly scattered over the whole ~655 MB cache whatever order the page table is walked in. Contiguity only helps the 4 B/token page-table reads (~1.5% of bytes). The 2.94 -> 2.67 TB/s drop is more plausibly a TLB-reach or DRAM-page effect of the doubled random footprint, which stream-K does not touch. Expect M1 on both shapes, M2 plausible at decode_16 (<=108 needs -11 us: ~5-7 from the ticket plus the tail), M2 at decode_32 (<=222, -32 us) unlikely (~-8 to -12 realistic), and M3 (<=1.5x floor, ~3.2 TB/s effective on random 128 B gathers) not reachable by this stage's changes. The milestones are readable from profile top_kernels, since v2 is a single named kernel replacing A+B. The expected directions match TASK.md: the full-layer weight of 16 applies only to decode_16/32, and every other shape is gated out.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv: the incumbent being replaced, not a confounder. v2 keeps its ring and math byte-identical, so a v2-vs-A+B delta is attributable to the split and epilogue only. The FINDINGS caveat applies: full decode_32 headline rows have shown a scoring artifact before (full-verify-cuda-streamk-splitkv 'gained' 1.14x with an unchanged kernel A), so judge decode_32 by kernel time.
- fused-preattention-triton: _pre_attention_kernel produces the FP8 q that v2 consumes, unchanged. It is a launch-sequence neighbour only. With B gone, v2 is directly followed by the graph end, so no interaction is expected.
- paired-heads-launch-cleanup / fa3-hdim128-head-pairing / ab-tune-batch-gate: the pairing predicate already excludes splitkv shapes ('and not splitkv'), so they are not on this path and do not confound.
- sliding-deep-splits-on-cuda-splitkv / sliding-fused-triton-decode1-gate: sliding-only and gated out of the full decode path. Their shapes should be flat; any headline movement there is noise.

### Interference and A/B needs
No companion is warranted. The spec already builds the needed controls in-process at import time: eager timings at synthetic B=16 and B=32 x 80000 of stage-1 A+B, stream-K without the ticket (+B), stream-K with the ticket, and 5 stages. Those separate the two sub-changes (work split vs epilogue fold) and the stages knob on the same GPU, in the same run, with the same inputs, which a separate okc run could not do more cleanly. No landed hypothesis shares this kernel's code path except the incumbent it replaces, so an ab-revert has nothing to isolate. For a rewrite, the next stage is the follow-up. Ask the builder to print those diagnostics as per-kernel µs and TB/s (bytes = sum(seq_b) x 256 B), so the decode_32 locality claim is directly confirmed or refuted by the 'stream-K no ticket' vs 'stage-1 A' line.

### Open questions
- Is the decode_32 bandwidth drop (2.94 -> 2.67 TB/s) a TLB-reach effect of the 2x random footprint? If so, stream-K cannot fix it, and stage 3 should instead test request/tile ordering that keeps concurrently active CTAs within a bounded slice of the cache (impossible with a randperm page table), or accept ~2.7 TB/s and stop at M2.
- Does the last-arriver merge at B=16 (~17 slots per request on one CTA) add a visible serialized tail? Compare 'stream-K with ticket' vs 'stream-K no ticket + B' in the diagnostics.
- splitkv-full-line-gather-mapping (unproven-vs-best at 1.416x, gained on full decode_16/32) changes the same issue_tile mapping that v2 keeps 'exactly'. Should stage 3 stack it onto v2, since v2 freezes the stage-1 gather?
- Two load_inline modules (v1 fallback + v2) at about 45 s each, plus eager diagnostics at B=32 x 80000 for 4 variants: confirm the total stays well inside the 15-minute budget with the 16 tests.


## Implementation notes

Built 2026-10-25 in the worktree (`muse_attention_decode/submission.py` only; not compiled here, no nvcc/GPU on this node).

- **New module `muse_full_decode_splitkv_v2`** (`_SK_CPP`/`_SK_CUDA`, built by `_build_sk()` with `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored after, `-O3 -lineinfo`; torch's resolved arch flags are printed). Stage-1 module v1 (kernels A + B), its build, its self-check and its code are untouched and stay the fallback; `_full_decode_splitkv` gains only a 2-line dispatch to `_full_decode_sk` when `_EXT2` is set.
- **`splitkv_attn_sk_kernel<STAGES, TICKET>`**: ring, swizzle, cp.async.cg K/V issue, m16n8k16 QK/PV math, online softmax and the 4-warp smem merge are copied verbatim from kernel A. Changed: (1) stream-K over 64-token tiles, CTA c owns [c*T/grid, (c+1)*T/grid) (int64 math), tile prefix rebuilt per CTA in smem; producer and consumer walk request boundaries with incremental cursors; page indices prefetched 16 tiles per block, double-buffered, 4-byte cp.async zero-filled past each request end (binary search for the request per block). (2) Segment end: a whole-request segment finalizes in place; otherwise the fp32 partial goes to slot c + b (workspace [grid + B] slots, allocated per call from grid and batch), `__threadfence`, ticket `atomicAdd`; the CTA drawing `c_hi - c_lo` (owner(t) = ((t+1)*grid - 1)/T, the exact inverse) fences, LSE-merges slots c_lo..c_hi via `__ldcg`, times v_scale, applies the stage-1 B epilogue (scaleless RMSNorm over 64, eps 1e-5 inside the sqrt, sigmoid(gate), bf16) and resets `ticket[b] = 0`. No CTA waits on another, so there is no spin to bound. Ticket: int32[64] per device, zeroed once at import (`_sk_ticket`).
- **Variants** (one template, three instantiations): 0 = <4 stages, ticket> (the only timed path), 1 = <4, no ticket> + `splitkv_sk_combine_kernel` (diagnostics), 2 = <5 stages, ticket> (112,672 B smem, still 2 CTA/SM by arithmetic; occupancy printed).
- **Import sequence**: v1 build + v1 self-check (master) -> v2 build -> `_sk_validate()`: device capability, per-variant `cudaFuncGetAttributes` binaryVersion/ptxVersion/regs/local bytes/smem and occupancy, arch flags; anything not 90/sm_90a turns v2 OFF -> canary (B=2, lengths 5 and 67) -> 8 self-check cases (B=2 one request spanning every CTA at grid 7 and device; B=16 ragged with seq % 64 in 1..7 and lengths < 64, kv_scale 0.25/1, grids device/3; B=32 ragged grids 3/device; B=64 mixed grids device/7) -> kernel_fn captured as a CUDA graph and replayed twice with the output NaN-poisoned before each replay -> tickets must sum to 0. Every launch is announced with a flushed print, synced, and confirmed. Any failure prints `PATH CHOSEN: full decode splitkv v1 (fallback: <reason>)`; success prints `PATH CHOSEN: full decode streamk+ticket v2`. A final line names the unchanged path for every other shape.
- **Diagnostics (review's request, never choose the path)**: after PATH CHOSEN, variants 1 and 2 are correctness-checked (skipped from timing if they fail or cannot launch), then at synthetic B=16 and B=32 x 80000 (randperm page table, as reference.py) eager cold-L2 (256 MB flush) median-of-10 per-kernel µs and TB/s (bytes = sum(seq_b) x 256) for stage-1 A, stage-1 B, stream-K + ticket (4 st), stream-K no ticket + its combine, stream-K + ticket (5 st), each with rel L2 vs the stage-1 output, plus a ratio line: `stage-1 A / stream-K no-ticket A` (the decode_32 locality claim, confirm/refute), `(no-ticket A + combine) / ticket` (the fold and the last-arriver tail), `4 st / 5 st`. Read decode_32 from these lines and profile top_kernels, not the headline.
- **Deviations from the spec**: (a) the next request's query is not fetched into "a second smem slot when a segment starts": it rides the segment's first tile into a per-stage 1 KB query slot (STAGES slots), which keeps it in the same cp.async group as the tile and needs no extra wait. (b) Since segments are contiguous, every CTA in c_lo..c_hi has a non-empty intersection with the request, so there are no empty segments; merge still weights -inf partials 0. (c) Per the review, the decode_32 M2 target is treated as doubtful (randperm page table: contiguity only helps the 4 B/token index reads), hence the diagnostic ratio lines.
- Review fixes applied: per-kernel µs and TB/s diagnostics with the named comparison lines (the only required fix); all build/safety rules from LESSONS 2026-10-25 (in-process only, sm_90a only, -lineinfo, plain cp.async.cg, sync+print per launch). First run must be `okc check`.

## Rules check

Checked against `git -C .worktrees/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc diff`.

- [x] CUDA-graph capturable: `_full_decode_sk`/`_sk_launch` do only `torch.empty` from shapes/grid and two extension launches; no `.item()`/sync/print in kernel_fn's call graph (lint: all host-sync hits are outside it); branches only on `_EXT2`/`_SK_VARIANT` module state and config; the self-check captures and replays kernel_fn itself.
- [x] Nothing derived from `data` cached: the ticket is a zero int32[64] keyed by device index, allocated at import, and reset to 0 by every launch; workspaces are per call.
- [x] Weights: n/a: this problem has no weights.
- [x] Activations BF16 where SGLang keeps BF16 (bf16 output, gate read as bf16, fp32 accumulation/epilogue); KV cache FP8 read as E4M3 widened exactly to f16; queries FP8 E4M3 (unchanged pre-attention kernel).
- [x] Every key attended: tiles cover [0, cache_seqlens[b]) of every request exactly once across CTAs; masking only past the length; no skipping by weight.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses (none: no subprocess/multiprocessing/fork/spawn anywhere) or network. The `TORCH_CUDA_ARCH_LIST` env var is set only around our own `load_inline` and restored.
- [x] All work inside the timed call; no output caching or replay (the import-time graph is our own synthetic self-check, deleted after).
- [x] Every other shape keeps master code byte for byte (the only edits outside new code are the 2-line dispatch inside the gated `_full_decode_splitkv`, the module docstring and two imports).

okc_lint output:

```
warn  host-sync    L1722: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1725: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1730: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1749: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1756: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1761: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1816: left = int(_sk_ticket(dev).abs().sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1850: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1861: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

py_compile: OK; `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (reading only, plus a Python emulation of the tile split; no GPU here). **Verdict: dispatch-with-fixes.**

Fixes applied in the worktree `submission.py`:
1. **The gate could never pass (category 1).** When Ttot < grid, some CTAs own zero tiles. The CTAs in `c_lo..c_hi` then include empty ones. The ticket target `c_hi - c_lo` was never reached, nothing finalized the request, the ticket was left non-zero, and `merge_slots` would have read slots that were never written. Implementation note (b) is false in this case. The canary (lengths [4, 66] -> cache_seqlens 5, 67 -> Ttot = 3 at grid 264) hits it every time, so v2 would always have fallen back to v1. Fix: `splitkv_attn_sk_kernel` now uses the effective grid `gx = min(gridDim.x, Ttot)` and CTAs at or above it exit. `splitkv_sk_combine_kernel` clamps `attn_grid` the same way. An emulation over 20k random (B, lengths, grid) cases, the canary included, confirms that every CTA in `c_lo..c_hi` then holds exactly one segment of the request. The scored shapes (Ttot around 20k-40k tiles) are unaffected.
2. **The diagnostics could poison the timed path (category 2/6).** Variant 2 (5 stages, ticket) shares the ticket buffer with the timed variant 0, and its correctness check and timing run after PATH CHOSEN. A failing variant-2 run could leave tickets non-zero for the timed path. Fix: the tickets are re-zeroed and synced in a `finally` after `_sk_diagnostics()`.

Checked and found sound: the ticket protocol (every thread runs `__threadfence` after writing its partial, then a barrier, then the atomic; the last arriver fences and reads with `__ldcg`). The reset needs no protection: it happens after every increment for that request, and no PDL attribute is used, so consecutive launches and graph replays are stream-ordered. The per-stage query slot has no RAW hazard (same cp.async group as its tile, then wait and barrier) and no WAR hazard (the slot is rewritten only after the next iteration's barrier, and its contents are already in registers). Slots `c + b` are unique and fit in `grid + B`. The tile-owner inverse is exact. Every key is attended once and masking happens only past the length. The FP8 query and KV and the bf16 output are unchanged. Nothing runs as a child process, and kernel_fn is capturable. okc_lint: OK with warnings (host syncs at import only); py_compile: OK.

Residual risk: never compiled. Register pressure under the 128-register cap (`__launch_bounds__(256, 2)`) and spills are unverified (the import log prints regs and local bytes), and the ptxVersion==90 gate relies on cubin-only images reporting the virtual arch.

## Result

- job `e592b1f5d3db4f83` (check), status done, passed True, tests 16/16
- speedup 1.4171x, 95% CI [1.3958, 1.4426], score 2657.769844158961 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-splitkv-s2-streamk-ticket-inproc.patch`; branch `hyp/muse_attention_decode/full-decode-splitkv-s2-streamk-ticket-inproc`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.17 | 13.09 | 0.9799 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 27.26 | 27.19 | 1.0058 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.34 | 15.19 | 1.0046 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 122.79 | 110.91 | 1.1047 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.24 | 24.29 | 0.9878 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 241.9 | 226.81 | 1.0661 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 16.72 | 16.83 | 0.9979 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.03 | 29.84 | 1.0004 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.67 | 22.72 | 0.9978 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 161.98 | 163.42 | 0.9897 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.7 | 28.87 | 0.9911 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 300.83 | 302.06 | 0.9956 | flat |

gained on full decode_16, decode_32. Expected hit: 2/3. Noise ±1.7%.

## Follow-ups

- rewrite stage 2 of `full-layer decode at 1<batch<=64: splitkv_attn_kernel A + splitkv_combine_kernel B (stage 1, master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode-splitkv-s2-streamk-ticket-inproc --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode-splitkv-s2-streamk-ticket-inproc`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

