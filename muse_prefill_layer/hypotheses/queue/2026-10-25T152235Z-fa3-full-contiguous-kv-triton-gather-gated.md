---
problem: muse_prefill_layer
slug: fa3-full-contiguous-kv-triton-gather-gated
created: 2026-10-25T152235Z
base: master
area: "FA3 call on full-attention layers (KV source: contiguous TMA copy of the paged page-size-1 rows, per-class timing gate)"
change: >
  Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention
  layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 changes. Sliding layers, the
  page-64 and kv-scaled cases, qkv GEMM, FusedQKNorm, the q/k/v FP8 casts, store_cache,
  o_norm_gate and everything after stay master's. MECHANISM: at page_size 1 FA3 runs its paged
  non-TMA path: producer warps gather KV rows with cp.async, PackGQA is forced and the key tile is
  smaller. fa3-contiguous-kv-gather (job 75bbf278dd9e430e, regression 0.988x) measured FA3's own
  kernel on a contiguous cache at 857 / 4300 / 7747 us on first_turn full ctx 0 / 32768 / 65536,
  against 907 / 4545 / 8164 paged in the same-era parity run (-5 to -6%). It also measured 1563 vs
  1635 us at turn_4 full and 405 vs 419 us at turn_1 full. It lost for two reasons. Its torch
  advanced-index gather (k_cache.view(-1,2,64)[page_table.clamp(min=0)],
  at::native::vectorized_gather_kernel) cost 100.5 us at turn_1 and 391.6 us at turn_4, about 0.2
  TB/s. And it was applied to every shape unconditionally, sliding included. This hypothesis keeps
  the TMA kernel win and removes both losses. (1) GATHER: a new Triton kernel
  _kv_gather_contig(k_cache, v_cache, page_table, cache_seqlens, k_buf, v_buf, W) with grid (B,
  cdiv(W, 128)). Each program loads 128 page_table entries of request b and widens them to int64
  rows. For each position p < cache_seqlens[b] it copies the 128-byte K row and the 128-byte V row
  (both kv heads, viewed as uint8 [rows, 128], 16-byte vector loads) into k_buf[b, p] and v_buf[b,
  p]. For p >= cache_seqlens[b] it stores zeros, so a masked tail tile never feeds FP8 NaN bytes
  into P x V. W = page_table.shape[1] * page_size, a shape. k_buf and v_buf are torch.empty((B, W,
  2, 64), float8_e4m3fn) per call: shape-derived, graph-safe, no module-global buffers. The kernel
  runs after store_cache, so the step's own rows are read back exactly as FA3 reads them today.
  (2) CALL: flash_attn_with_kvcache(q=q_fp8, k_cache=k_buf, v_cache=v_buf, page_table=None,
  cache_seqlens=data['cache_seqlens'], causal=True, the same softmax_scale, window_size (-1, -1),
  scheduler_metadata=None), in one of two modes. VARLEN keeps cu_seqlens_q and max_seqlen_q as
  master does. BATCHED passes q.view(B, L, 16, 64) with no cu_seqlens_q. That is legal because
  every prefill config has the same q_len for each request. BATCHED is the form the
  fa3-batched-mode review asked to re-mint on a contiguous base. The output is reshaped to [T,
  1024] for o_norm_gate exactly as now. (3) PREPARE-TIME HARD GATE: for each full-layer class the
  judge scores (turn_1 b1 L753 ctx~80000; turn_4 b4 L753 ctx~80000; first_turn b1 L16384 at ctx 0,
  32768 and 65536), build synthetic FP8 caches with a randomly permuted page-size-1 page table at
  the scored widths. CUDA-graph capture three candidates on the same pre-cast FP8 q: A = master's
  paged call; B = gather + contiguous VARLEN; C = gather + contiguous BATCHED. Before every replay
  write a flush buffer of at least 2x L2. Alternate the order, and take the median of at least 25
  replays. Route the class to the faster of B and C only if it is <= 0.97 x A; otherwise A. Key
  the decision by config-only fields, as attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc's
  _full_attn_route did: (batch, q_len, round((page_table width - q_len) / 16384)). Any key not
  timed runs A. Log '[fa3 contig] GATE <class>: paged X us, contig-varlen Y us, contig-batched Z
  us -> A|B|C', with the gather's own us from a separate capture, and one 'PATH CHOSEN' line per
  routed shape the first time kernel_fn sees it (a de-dup set used for printing only, never for
  routing). (4) SELF-CHECK in prepare(), before the gate: B and C against A on coherent synthetic
  data (keys sharing a direction, so softmax is peaked, per LESSONS 2026-10-25 on FP8 P) for b1
  ctx3000 L753, b4 seq_lens {0, 777, 5000, 80000} L753, b1 ctx0 L16384 and b2 {37, 1000} L211.
  Pass: rel L2 <= 1e-2, worst row <= 5e-2, all finite. A failure turns that variant off. Any
  exception anywhere in (1)-(4) leaves every class on A, so master's path runs. kernel_fn gains
  only two torch.empty calls, one Triton launch and the FA3 call: no host sync, no branching on
  values, and nothing derived from data is cached.
expected:
  - shapes: "*ctx=65536*layer=full*load=first_turn*"
    direction: faster
    why: FA3 kernel 8352.6 us of 18349.9 us (45%). The contiguous run measured -5.1% on FA3 (8164 -> 7747); the gather moves ~42 MB (~15-20 us). Net ~-400 to -600 us, 2-3% of the shape. Read the FA3 row and the GATE line.
  - shapes: "*ctx=32768*layer=full*load=first_turn*"
    direction: faster
    why: FA3 4542.4 us of 14825.7 (30%). -5.4% measured on contiguous (4545 -> 4300); the gather is ~25 MB. Net ~-220 us, ~1.5% of the shape.
  - shapes: "*ctx=0*layer=full*load=first_turn*"
    direction: flat
    why: FA3 911.7 us of 11599.6; -5% is ~45 us, 0.4% of the shape, inside noise.
  - shapes: "*layer=full*load=turn_4*"
    direction: flat
    why: FA3 1643.1 us. The contiguous kernel saved ~72 us, but a ~165 MB gather (~50 us at ~3.3 TB/s) eats most of it. The gate decides; at best ~-25 us (0.7%).
  - shapes: "*layer=full*load=turn_1*"
    direction: flat
    why: FA3 417.5 + 6.9 combine. A -14 us kernel saving against a ~12 us gather; the gate most likely keeps the paged path.
  - shapes: "*layer=sliding*"
    direction: flat
    why: "Sliding layers keep master's paged FA3 call unchanged."
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-full-contiguous-kv-triton-gather-gated

## Change

Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 changes. Sliding layers, the page-64 and kv-scaled cases, qkv GEMM, FusedQKNorm, the q/k/v FP8 casts, store_cache, o_norm_gate and everything after stay master's. MECHANISM: at page_size 1 FA3 runs its paged non-TMA path: producer warps gather KV rows with cp.async, PackGQA is forced and the key tile is smaller. fa3-contiguous-kv-gather (job 75bbf278dd9e430e, regression 0.988x) measured FA3's own kernel on a contiguous cache at 857 / 4300 / 7747 us on first_turn full ctx 0 / 32768 / 65536, against 907 / 4545 / 8164 paged in the same-era parity run (-5 to -6%). It also measured 1563 vs 1635 us at turn_4 full and 405 vs 419 us at turn_1 full. It lost for two reasons. Its torch advanced-index gather (k_cache.view(-1,2,64)[page_table.clamp(min=0)], at::native::vectorized_gather_kernel) cost 100.5 us at turn_1 and 391.6 us at turn_4, about 0.2 TB/s. And it was applied to every shape unconditionally, sliding included. This hypothesis keeps the TMA kernel win and removes both losses. (1) GATHER: a new Triton kernel _kv_gather_contig(k_cache, v_cache, page_table, cache_seqlens, k_buf, v_buf, W) with grid (B, cdiv(W, 128)). Each program loads 128 page_table entries of request b and widens them to int64 rows. For each position p < cache_seqlens[b] it copies the 128-byte K row and the 128-byte V row (both kv heads, viewed as uint8 [rows, 128], 16-byte vector loads) into k_buf[b, p] and v_buf[b, p]. For p >= cache_seqlens[b] it stores zeros, so a masked tail tile never feeds FP8 NaN bytes into P x V. W = page_table.shape[1] * page_size, a shape. k_buf and v_buf are torch.empty((B, W, 2, 64), float8_e4m3fn) per call: shape-derived, graph-safe, no module-global buffers. The kernel runs after store_cache, so the step's own rows are read back exactly as FA3 reads them today. (2) CALL: flash_attn_with_kvcache(q=q_fp8, k_cache=k_buf, v_cache=v_buf, page_table=None, cache_seqlens=data['cache_seqlens'], causal=True, the same softmax_scale, window_size (-1, -1), scheduler_metadata=None), in one of two modes. VARLEN keeps cu_seqlens_q and max_seqlen_q as master does. BATCHED passes q.view(B, L, 16, 64) with no cu_seqlens_q. That is legal because every prefill config has the same q_len for each request. BATCHED is the form the fa3-batched-mode review asked to re-mint on a contiguous base. The output is reshaped to [T, 1024] for o_norm_gate exactly as now. (3) PREPARE-TIME HARD GATE: for each full-layer class the judge scores (turn_1 b1 L753 ctx~80000; turn_4 b4 L753 ctx~80000; first_turn b1 L16384 at ctx 0, 32768 and 65536), build synthetic FP8 caches with a randomly permuted page-size-1 page table at the scored widths. CUDA-graph capture three candidates on the same pre-cast FP8 q: A = master's paged call; B = gather + contiguous VARLEN; C = gather + contiguous BATCHED. Before every replay write a flush buffer of at least 2x L2. Alternate the order, and take the median of at least 25 replays. Route the class to the faster of B and C only if it is <= 0.97 x A; otherwise A. Key the decision by config-only fields, as attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc's _full_attn_route did: (batch, q_len, round((page_table width - q_len) / 16384)). Any key not timed runs A. Log '[fa3 contig] GATE <class>: paged X us, contig-varlen Y us, contig-batched Z us -> A|B|C', with the gather's own us from a separate capture, and one 'PATH CHOSEN' line per routed shape the first time kernel_fn sees it (a de-dup set used for printing only, never for routing). (4) SELF-CHECK in prepare(), before the gate: B and C against A on coherent synthetic data (keys sharing a direction, so softmax is peaked, per LESSONS 2026-10-25 on FP8 P) for b1 ctx3000 L753, b4 seq_lens {0, 777, 5000, 80000} L753, b1 ctx0 L16384 and b2 {37, 1000} L211. Pass: rel L2 <= 1e-2, worst row <= 5e-2, all finite. A failure turns that variant off. Any exception anywhere in (1)-(4) leaves every class on A, so master's path runs. kernel_fn gains only two torch.empty calls, one Triton launch and the FA3 call: no host sync, no branching on values, and nothing derived from data is cached.

## Expected

- `*ctx=65536*layer=full*load=first_turn*` faster: FA3 kernel 8352.6 us of 18349.9 us (45%). The contiguous run measured -5.1% on FA3 (8164 -> 7747); the gather moves ~42 MB (~15-20 us). Net ~-400 to -600 us, 2-3% of the shape. Read the FA3 row and the GATE line.
- `*ctx=32768*layer=full*load=first_turn*` faster: FA3 4542.4 us of 14825.7 (30%). -5.4% measured on contiguous (4545 -> 4300); the gather is ~25 MB. Net ~-220 us, ~1.5% of the shape.
- `*ctx=0*layer=full*load=first_turn*` flat: FA3 911.7 us of 11599.6; -5% is ~45 us, 0.4% of the shape, inside noise.
- `*layer=full*load=turn_4*` flat: FA3 1643.1 us. The contiguous kernel saved ~72 us, but a ~165 MB gather (~50 us at ~3.3 TB/s) eats most of it. The gate decides; at best ~-25 us (0.7%).
- `*layer=full*load=turn_1*` flat: FA3 417.5 + 6.9 combine. A -14 us kernel saving against a ~12 us gather; the gate most likely keeps the paged path.
- `*layer=sliding*` flat: Sliding layers keep master's paged FA3 call unchanged.

## Risk

Low risk: the same FA3 binary, the same FP8 q and KV and P precision, and every key attended; only the KV source changes from paged rows to a contiguous copy of the same rows. Headline stake is small (~+0.3%; first_turn load +0.6%, turn_4 at most +0.3%), so judge it on the FA3 row per routed shape and the GATE lines. The scored effect may sit inside the +-1.5% interval. Distinct from the regressed fa3-contiguous-kv-gather in form, which its follow-up forbade repeating: a bandwidth-sane Triton gather instead of torch indexing, full layers only, a per-class cold-L2 timing gate instead of unconditional routing, and a zeroed tail. Failure modes: (a) FA3's contiguous path may pick a tile config whose advantage is smaller at the judge's widths than at the parity era's: the gate makes this flat, not a loss; (b) the batched-mode variant may fail to accept FP8 q without cu_seqlens_q in sgl_kernel's wrapper: caught by the self-check, which turns C off. The step's rows must be gathered after store_cache (same stream). If it lands, it is the base a later stage can use to put the prefix gather on a side stream under the qkv GEMM (prefix rows do not depend on this call's qkv), which would let turn_4 route too.
