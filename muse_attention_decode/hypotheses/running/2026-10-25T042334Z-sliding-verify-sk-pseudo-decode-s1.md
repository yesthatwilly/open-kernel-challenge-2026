---
problem: muse_attention_decode
slug: sliding-verify-sk-pseudo-decode-s1
created: 2026-10-25T042334Z
base: master
area: sliding-layer attention (verify, stream-K pseudo-decode kernel)
change: >
  Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode
  stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64 and the two-level ticket
  at B=1, landed as full-decode1-streamk-s3-two-level-ticket, c3c1b01). Idea: a verify block of 8
  tokens is 8 decode queries, each over its own 2048-key window, so route SLIDING verify (window
  != 0, q_len 8, tp8, head_dim 64, page 1, 1<=B<=32; config-only predicate) through a template
  variant of the stream-K decode kernel over P = 8B pseudo-requests, keeping its gather-bound
  16-row (2 kv heads x 8 q heads) mma.sync design, instead of the 64-row verify tiles that LESSONS
  says are MMA-bound. Kernel edit, new template flag PSEUDO (false = the landed decode code,
  byte-identical): pseudo-request p in [0, 8B) maps to request b = p / 8 and token j = p % 8;
  seq_p = cache_seqlens[b] - 7 + j (keys 0..seq_p-1, so causality inside the block is exact), lo_p
  = max(0, seq_p - window) (matches FA3 window_size=(window-1, 0) as master passes it; every key
  in each window attended), tiles_p = ceil((seq_p - lo_p) / 64) with tile k covering keys lo_p +
  64k .. +63 (page 1, no alignment needed; mask t < seq_p on the last tile). The tile prefix over
  P <= 256 pseudo-requests is built in smem from cache_seqlens exactly as the decode kernel builds
  its per-request prefix (tp_s grows to 257 entries). Page lookup uses page_table row b; the FP8 q
  block is token b*8+j of the q that master _pre_attention_kernel already writes (PAIRED=False,
  ROPE on; it also stores the 8 new K/V rows, so the kernel needs no rope and no new-key build);
  the ticket epilogue norm-gates with qkv row b*8+j and stores out row b*8+j; workspace slots grid
  + P; ticket counters sized for P. Reuse is from L2: consecutive pseudo-requests share 2041 of
  2048 keys and stream-K puts them on neighbouring CTAs at the same time, so DRAM bytes stay ~1x
  (8.4 MB at B=16) while L2 reads are 8x. Hard gate in prepare (LESSONS: PATH CHOSEN must compare
  printed timings): per band B=1, B=2..16, B=17..32 time (cold L2, median of 10, CUDA events) the
  pseudo-decode kernel (single-level and two-level ticket variants, grid SM and 2xSM) against
  master FA3 path (prepare + main + combine + norm-gate, pre-attn excluded from both), print every
  number, and route a band only if its best variant is >= 5% faster; otherwise that band keeps
  FA3. Self-check against an fp32 windowed causal reference: window 2048 with lens < 8, < window,
  < window + 8 and 80000, B = 1/3/16/32, kv_scale 1 and 0.25, grids 3/7/device; on failure only
  this route is disabled (PATH CHOSEN: sliding verify FA3 (fallback: <case, rel L2>)).
  BUILD/SAFETY as the landed kernel (TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, -lineinfo,
  in-process canary, sync+print after each kernel, never a child process). Sliding decode, full
  layers and page 64 untouched. First measurement is an okc check.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: replaced ~14.3 us (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate ~1.1) of a 16.7 us shape, floor 0.1; 8 pseudo x 32 tiles = ~264 tiles = one tile per CTA at grid 2xSM plus one ticket merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: replaced ~17 us (FA3 10.7 + combine 4.1 + prepare + norm-gate) of 22.7; 67 MB of L2 reads over 8.4 MB of DRAM. M2 shape <= 18 us
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: replaced ~21 us of 28.7; 134 MB L2 reads may make this band lose, then the hard gate keeps FA3 (flat)
  - shapes: "*load=decode_* *layer=full*"
    direction: flat
    why: PSEUDO=false instantiation is the landed decode code; full and decode paths untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate"
review: proceed
status: running
review_at: 2026-10-25T043142Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T043142Z
base_sha: 4488c6a7868d7d953811b7b93ec0403d0f8bfde8
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1
worktree: .worktrees/muse_attention_decode/sliding-verify-sk-pseudo-decode-s1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T044314Z
impl_commit: 6f95ce71ba21682ba9cbb6d346abb2a93a862b2f
submission_sha256: e1e1054ff99e9be0e1cb0f44d06c07cc54de33d08197c48e72668db0f6b93a40
norm_sha256: a37cfcac0e6a83aff97b999dbdbbd7e7c7676af1abccc429ae2d61b928d3e156
submission_blob: 76492911dfd6685139a2ffac06c1d27083ecdb0a
lint_warnings: 27
attempt_file: 2026-10-25T044314Z-check-sliding-verify-sk-pseudo-decode-s1.md
prepared_at: 2026-10-25T044314Z
job: 94523c3313444fd8
dispatched_at: 2026-10-25T044328Z
---

# sliding-verify-sk-pseudo-decode-s1

## Change

Stage 1 of a new sliding-verify line, built on master (which now carries the full-decode stream-K kernels: splitkv_attn_sk_kernel single-level ticket at 1<B<=64 and the two-level ticket at B=1, landed as full-decode1-streamk-s3-two-level-ticket, c3c1b01). Idea: a verify block of 8 tokens is 8 decode queries, each over its own 2048-key window, so route SLIDING verify (window != 0, q_len 8, tp8, head_dim 64, page 1, 1<=B<=32; config-only predicate) through a template variant of the stream-K decode kernel over P = 8B pseudo-requests, keeping its gather-bound 16-row (2 kv heads x 8 q heads) mma.sync design, instead of the 64-row verify tiles that LESSONS says are MMA-bound. Kernel edit, new template flag PSEUDO (false = the landed decode code, byte-identical): pseudo-request p in [0, 8B) maps to request b = p / 8 and token j = p % 8; seq_p = cache_seqlens[b] - 7 + j (keys 0..seq_p-1, so causality inside the block is exact), lo_p = max(0, seq_p - window) (matches FA3 window_size=(window-1, 0) as master passes it; every key in each window attended), tiles_p = ceil((seq_p - lo_p) / 64) with tile k covering keys lo_p + 64k .. +63 (page 1, no alignment needed; mask t < seq_p on the last tile). The tile prefix over P <= 256 pseudo-requests is built in smem from cache_seqlens exactly as the decode kernel builds its per-request prefix (tp_s grows to 257 entries). Page lookup uses page_table row b; the FP8 q block is token b*8+j of the q that master _pre_attention_kernel already writes (PAIRED=False, ROPE on; it also stores the 8 new K/V rows, so the kernel needs no rope and no new-key build); the ticket epilogue norm-gates with qkv row b*8+j and stores out row b*8+j; workspace slots grid + P; ticket counters sized for P. Reuse is from L2: consecutive pseudo-requests share 2041 of 2048 keys and stream-K puts them on neighbouring CTAs at the same time, so DRAM bytes stay ~1x (8.4 MB at B=16) while L2 reads are 8x. Hard gate in prepare (LESSONS: PATH CHOSEN must compare printed timings): per band B=1, B=2..16, B=17..32 time (cold L2, median of 10, CUDA events) the pseudo-decode kernel (single-level and two-level ticket variants, grid SM and 2xSM) against master FA3 path (prepare + main + combine + norm-gate, pre-attn excluded from both), print every number, and route a band only if its best variant is >= 5% faster; otherwise that band keeps FA3. Self-check against an fp32 windowed causal reference: window 2048 with lens < 8, < window, < window + 8 and 80000, B = 1/3/16/32, kv_scale 1 and 0.25, grids 3/7/device; on failure only this route is disabled (PATH CHOSEN: sliding verify FA3 (fallback: <case, rel L2>)). BUILD/SAFETY as the landed kernel (TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, -lineinfo, in-process canary, sync+print after each kernel, never a child process). Sliding decode, full layers and page 64 untouched. First measurement is an okc check.

## Expected

- `*layer=sliding load=verify_1 *` faster: replaced ~14.3 us (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate ~1.1) of a 16.7 us shape, floor 0.1; 8 pseudo x 32 tiles = ~264 tiles = one tile per CTA at grid 2xSM plus one ticket merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
- `*layer=sliding load=verify_16*` faster: replaced ~17 us (FA3 10.7 + combine 4.1 + prepare + norm-gate) of 22.7; 67 MB of L2 reads over 8.4 MB of DRAM. M2 shape <= 18 us
- `*layer=sliding load=verify_32*` faster: replaced ~21 us of 28.7; 134 MB L2 reads may make this band lose, then the hard gate keeps FA3 (flat)
- `*load=decode_* *layer=full*` flat: PSEUDO=false instantiation is the landed decode code; full and decode paths untouched

## Risk

Kill: if the prepare timing shows the pseudo-decode kernel slower than the FA3 path at B=1 (the most latency-bound band), the line is dead. Risk: 8x L2 read amplification at B=32; tp_s smem growth; correctness at the window lower edge (tests put keys at window edges).

## Milestones

Rewrite stage 1 of `sliding verify (q_len 8, page 1, tp8, 1<=B<=32): prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T043142Z by an adversarial subagent (model opus).

**Verdict: proceed.** The causal and window algebra is correct: seq_p = cache_seqlens[b]-7+j = p+1 and lo_p = seq_p-W reproduce the reference's keys[max(0,p-W+1):p+1] exactly. The per-band >=5% gate keeps every result attributable. Expect the win, if any, only at B=1, since 8x L2 read amplification makes B=16/32 unlikely. Add small-grid variants so the per-pseudo-request merge is a few slots, not ~32.

### Justification
Mechanism: treating each verify token as a decode query keeps the gather-bound 16-row (2 kv heads x 8 q heads) mma.sync tile, instead of the 64-row verify tiles that lost 1.6-2.1x on mma.sync (REWRITE_READ sections 2-4). At B=1 the target (sliding verify_1, 16.7 us shape, ~14.3 us replaced) is latency-bound: 4 MB of L2 reads over 0.5 MB of DRAM, 256 tiles. WEAKEST LINK: the B=1 merge. At grid 2xSM each pseudo-request's 32 tiles land on ~32 CTAs, so the last arrival reads ~32 x 4.2 KB slots serially. Stage 3's own diagnostic priced that at ~0.1 us/slot (single-level grid SM over 132 slots = 34.8 us vs two-level 23.1), so ~3-4 us of serial merge is on top of one gather round trip. The proposed variants (grid SM / 2xSM, single / two-level) all keep ~16-32 partials per pseudo-request. A grid of 4*P or 8*P CTAs (32/64 at B=1, so 4-8 tiles per CTA and 4-8 slots per merge) is the variant most likely to clear 13 us, and it is config-derived. B=16/32: 67/134 MB of L2-served cp.async. The landed kernel moves ~3 TB/s from DRAM, and even at 2x that from L2, 67 MB is ~11 us of kernel, against FA3 main 10.7 + combine 4.1. B=16 is a coin flip and B=32 likely gated to FA3, as the hypothesis anticipates. Milestones are readable: the sk kernel row replaces prepare_varlen + FA3 + combine + norm-gate on each sliding verify row.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket / full-decode-splitkv-s2 (c3c1b01): this is the kernel being templated. PSEUDO=false must be byte-identical. Print sk_attrs for every landed variant (regs/local/smem/CTAs per SM) and compare with the landed log (121 regs, 0 local, 95264 B, 2 CTAs/SM). tp_s growing to 257 ints adds ~1 KB smem, and if it is in the shared Lay<> it changes the landed instantiations' smem too. Size it per PSEUDO instantiation or show the landed numbers unchanged.
- fused-preattention-triton (landed): the q (PAIRED=False, ROPE on, FP8) and the 8 new cache rows come from it. Excluding pre-attn from both sides of the prepare timing is correct.
- sliding-verify1-triton-pseudo-decode and sliding-cluster-s3-verify-attn-only-phase-timed (queued, same rows): three implementations target sliding verify_1. This one is the most general (all bands, internal FA3 gate). Run it first. Each competitor is attributable only if it runs against a base where sliding verify is FA3, so never stack them; the comparison is by printed kernel timings.

### Interference and A/B needs
No companion. Each band is routed only if its printed best variant beats the printed FA3 time by >=5%, so every sliding verify row is either this kernel (named in PATH CHOSEN with its variant and grid) or byte-identical FA3. Sliding decode and full rows are untouched code. A gain or loss on any row is therefore attributable without an A/B run.

### Open questions
- BUILDER: add grid = 4*P and 8*P (config-derived: 32/64 CTAs at B=1, 512/1024 capped at 2xSM for B>=16) to the variant matrix, so a pseudo-request's merge is 4-8 slots. Print each variant's time, and the tile-to-CTA layout at B=1.
- BUILDER: order tiles so that pseudo-requests of the same b and the same key offset are co-resident (the request-major prefix already does this when P per request is ~equal). Print L2 hit rate or DRAM bytes for B=16 in the prepare if cheap, to confirm the ~1x DRAM assumption.
- If B=1 routes and wins, the held sliding-verify1-triton-pseudo-decode is redundant. If B=1 falls back to FA3, that Triton variant is the next thing to try on this row.


## Implementation notes

All edits in `muse_attention_decode/submission.py` (worktree; +591/-39). Nothing could be run on the build node (no GPU, torch or nvcc), so the CUDA is uncompiled.

- Kernel: `splitkv_attn_sk_kernel<STAGES, TICKET, TWO, PSEUDO=false>`. PSEUDO runs over R = 8B pseudo-rows r = 8b+j. A new `load_prefix_pseudo` (block-wide shuffle scan, one thread per row, so no 256-iteration serial prefix) fills cnt_s[r] = sp - lo (sp = cache_seqlens[b]-7+j, lo = max(0, sp-2048)), lo_s[r] and the tile prefix. The rest of the kernel treats cnt_s as the request length, so masks, tile counts and zero-fill are unchanged. `lo_s` is added only to the page-table offset in `issue_meta`, and the page-table row is r>>3. q, qkv (gate) and out rows are r, so `finalize_store` is untouched. find_req and T use R. The ticket, workspace and group slots are sized for R (ws grid+R, tickets 256 / 256*32 in separate `_ps_*` buffers).
- Review fix (landed layout): `Lay<STAGES, MAXR=64, WITH_LO=false>` reproduces the landed offsets exactly (BYTES unchanged); PSEUDO instantiations use `Lay<STAGES, 256, true>` (97824 B at 4 stages), so tp_s growth is not in the shared layout. The module is built with `-DSK_PSEUDO=1` first; if that build fails it is rebuilt without it (the landed kernels only) and sliding verify stays on FA3. `_sk_validate` prints the variant-0 regs/local/smem/CTAs against 121 / 0 / 95264 B / 2 and says IDENTICAL or DIFFERS; PSEUDO attrs are printed too. SASS identity for PSEUDO=false cannot be proven offline: the kernel signature is unchanged (window is the constant 2048, so the gate requires window == 2048) and the PSEUDO code is under `if constexpr`.
- Review fix (variants): single-level (4) and two-level (5) tickets x grid 4R, 8R, SM, 2xSM (all capped at 2xSM, deduplicated per band). At B=1 that is 32/64/132/264 CTAs (4-8 slots per merge at 4R/8R); the tiles-per-CTA and slots-per-pseudo-row layout is printed. DRAM bytes are printed analytically (no counter in prepare).
- Gate (`_ps_setup`, after stage-2/3 validation, before the diagnostics): canary + 18 self-check cases against the fp32 windowed causal reference (B=1/3/16/32; seq_lens 0, 3, 7, 1500, 2045, 2047, 2055, 80000; kv_scale 1/0.25; grids 3/7/4P/8P/SM/2xSM; both variants; every launch announced and synced). Per band B=1 / 16 / 32 x 80000 it re-checks each candidate against the reference, times candidates and the master FA3 path (prepare + main + combine + norm-gate, pre-attention excluded from both) as cold-L2 CUDA-graph replays (256 MB read before each, median of 10), prints every number, and routes a band (B=1, 2..16 from the B=16 timing, 17..32 from the B=32 timing) only if FA3/pseudo >= 1.05. Routed bands are then replay-checked through `kernel_fn` in a graph (NaN poison, two replays, tickets zero). `PATH CHOSEN` is printed per band.
- Deviations: no `-std=c++20` flag (the landed build does not pass one and passes no -gencode, so torch's default applies; adding it would change the landed build line). Sliding decode, full layers, page 64 and the FA3 call are untouched.
- Biggest risk: compile errors in the uncompiled CUDA (the retry without SK_PSEUDO bounds the damage to the landed kernels), then B=1 merge latency, which the 4R/8R grids target.

## Rules check

- [x] CUDA-graph capturable: `kernel_fn` adds only `_ps_launch` (torch.empty by shape, one extension launch); all `.item()`/synchronize calls are in import-time prepare code (lint: warnings only, outside kernel_fn).
- [x] Nothing derived from `data` is cached across calls: tickets and flush buffers are device-only allocations, zeroed once at import; every launch leaves tickets at 0.
- [x] Weights: n/a: no weights in this problem.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: the pre-attention kernel's e4m3 q is used, output bf16, fp32 softmax and merges.
- [x] Attention sees every key: each pseudo-row attends all keys max(0, p-2047)..p, causal per query; no skipping (checked against reference.py's `start = max(0, p - window + 1)`, `keys[start:p+1]`).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (grep: no subprocess/fork/multiprocessing; lint network errors were the word "requests" in comments and were reworded).
- [x] All work inside the timed call; no output caching or replay.

okc_lint output: `OK with warnings: submission.py` (host-sync warnings only, all in prepare-time code, outside kernel_fn). `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only reasoning; no GPU on the build node). **Verdict: dispatch-with-fixes.**

Checked and found sound:
- Window/causal algebra: cache_seqlens = seq_lens + 8 (reference.generate_input), so sp = cache_seqlens - 7 + j = seq_lens + j + 1 = p + 1 is the EXCLUSIVE end; keys [lo, sp) with lo = max(0, sp - 2048) = max(0, p - 2047) are exactly reference.py `keys[max(0, p - W + 1) : p + 1]`. No off-by-one; sp >= 1 always (cache_seqlens >= 8), so every pseudo-row has >= 1 tile and find_req/ticket invariants hold. Masks (`pos < cnt`) and page lookup (`page_table[b, lo + pos]`, page 1) are relative to lo consistently in issue_meta, issue_tile and the score mask.
- Rows: q token r = 8b + j matches the pre-attention [B*8, 16, 64] unpaired e4m3 query; qkv gate row r via the reshaped (tokens, -1) qkv stride; out [B, 8, 1024] row r; ws slot bx + r < grid + R; tickets 256 / 256*32 cover R <= 256, two-level groups <= 32. KV stays FP8, q is e4m3 (= FA3), fp32 softmax/merges, bf16 out.
- PSEUDO=false: Lay<STAGES, 64, false> reproduces the landed offsets and BYTES (95264 at 4 stages); R, prb/plo and the prefix are `if constexpr`/constant-folded, lo_s is never odr-used, so the landed variants 0-3 should compile to the same code; _sk_validate prints regs/local/smem/CTAs vs 121/0/95264/2.
- Build fallback: success compiles one module once (6 instantiations instead of 4, ~+30-50% of the ~40 s landed compile); only a failed -DSK_PSEUDO build triggers the landed-only rebuild. Within the 15 min budget.
- kernel_fn: config-only `_ps_gate`, shape-only torch.empty, one extension launch, no prints/try/except/host syncs; _PS_ROUTE is set once at import. Self-check reference matches reference.py (interleaved RoPE, FP8 rounding before qk_scale_factor, per-head output RMSNorm * sigmoid(gate)). Every failure path (build, self-check, gate, replay, raise) lands on byte-identical FA3 with a PATH CHOSEN line.

Fixes:
- `_ps_replay_check` now also REFILLS the captured inputs in place with different seq lens (same shapes; `_ps_synthetic(max_pages=...)`), replays the graph and checks against the reference of the refilled data, as the judge recheck does. The kernel reads cache_seqlens on device and sizes everything from config, so this should pass; it guards the routed band against a refill-only failure.

Residual risk: uncompiled CUDA (a compile error costs one extra landed-only build and leaves sliding verify on FA3; an in-kernel fault at import would poison the context for the whole run, as with the landed kernels).
