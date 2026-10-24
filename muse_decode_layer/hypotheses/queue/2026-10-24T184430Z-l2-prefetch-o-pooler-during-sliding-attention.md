---
problem: muse_decode_layer
slug: l2-prefetch-o-pooler-during-sliding-attention
created: 2026-10-24T184430Z
base: master
area: o GEMM + pooler GEMM (L2 warm-up overlapped with attention)
change: >
  On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.e.
  sliding decode_16, decode_32 and verify_1), overlap an L2 warm-up of the two dense weights the
  main stream reads right after attention with the attention phase itself. On these shapes the
  attention phase (FusedQKNorm .. FusedRMSNormGate, ~25-30 us) moves little DRAM traffic (a
  2048-token FP8 window, <= 8 MB), while the o GEMM and the pooler after it run near the HBM limit
  (pooler 67 MB in 20 us at T=16). New Triton kernel _l2_warm_kernel(ptr, n_words, sink_ptr,
  WORDS_PER_PROG, BLOCK: tl.constexpr = 4096): ptr is the weight viewed as int32 words; program p
  walks its contiguous WORDS_PER_PROG range in BLOCK steps with tl.load(ptr + offs, mask=offs <
  n_words, other=0, eviction_policy="evict_last"), XOR-accumulates into a register vector, and at
  the end stores the XOR-reduced scalar to sink_ptr + p only under the mask (acc == 0x7F3A5C1D) so
  the loads stay live; grid 32 programs (a quarter of the SMs, so attention keeps the rest).
  prepare(): state["prefetch_stream"] = torch.cuda.Stream(device); state["l2_sink"] =
  torch.zeros(64, dtype=torch.int32) on device (never read); flat int32 views of dense["o"] (16.8
  MB, whole) and of dense["pooler"][:1536] (rows 0..1535 = 25.2 MB), ~42 MB total, under the 50 MB
  L2. kernel_fn on the gated shapes: right after qkv = _gemm(...) (so the warm-up does not compete
  with the qkv weight stream), pf = state["prefetch_stream"]; pf.wait_stream(main); with
  torch.cuda.stream(pf): launch _l2_warm_kernel on the o view, then on the pooler slice; master
  attention continues on main unchanged. Join with main.wait_stream(pf) immediately before the
  return (the warm-up is long finished by then; this only closes the fork for graph capture). Only
  constant prepare-time weights are read; nothing derived from inputs is read or kept. Every other
  shape: master byte for byte.
expected:
  - shapes: "*layer=sliding load=decode_16 *"
    direction: faster
    why: "o (nvjet 64x16 2x1 pair with the unpooler, 26.1 us) and pooler (nvjet 128x16 splitK, 20.0 us) read 42 of their 84 MB from L2: milestone pooler row <= 15 us; -4..-8 us of 230 us, +2-3.5%"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: "same weights, pooler 21.2 us: -4..-8 us of 270 us, +1.5-3%"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "o 7.1 us + pooler half of the 128x8 splitK pair: -3..-6 us of 205 us, +1.5-3%"
  - shapes: "*layer=full*"
    direction: flat
    why: "untouched: full-layer attention streams 300+ MB of KV and would evict the warm lines"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate (restack changes those GEMMs)
  - shapes: "*load=verify_16 *"
    direction: flat
    why: T=128 outside the gate
risk: high
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# l2-prefetch-o-pooler-during-sliding-attention

## Change

On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.e. sliding decode_16, decode_32 and verify_1), overlap an L2 warm-up of the two dense weights the main stream reads right after attention with the attention phase itself. On these shapes the attention phase (FusedQKNorm .. FusedRMSNormGate, ~25-30 us) moves little DRAM traffic (a 2048-token FP8 window, <= 8 MB), while the o GEMM and the pooler after it run near the HBM limit (pooler 67 MB in 20 us at T=16). New Triton kernel _l2_warm_kernel(ptr, n_words, sink_ptr, WORDS_PER_PROG, BLOCK: tl.constexpr = 4096): ptr is the weight viewed as int32 words; program p walks its contiguous WORDS_PER_PROG range in BLOCK steps with tl.load(ptr + offs, mask=offs < n_words, other=0, eviction_policy="evict_last"), XOR-accumulates into a register vector, and at the end stores the XOR-reduced scalar to sink_ptr + p only under the mask (acc == 0x7F3A5C1D) so the loads stay live; grid 32 programs (a quarter of the SMs, so attention keeps the rest). prepare(): state["prefetch_stream"] = torch.cuda.Stream(device); state["l2_sink"] = torch.zeros(64, dtype=torch.int32) on device (never read); flat int32 views of dense["o"] (16.8 MB, whole) and of dense["pooler"][:1536] (rows 0..1535 = 25.2 MB), ~42 MB total, under the 50 MB L2. kernel_fn on the gated shapes: right after qkv = _gemm(...) (so the warm-up does not compete with the qkv weight stream), pf = state["prefetch_stream"]; pf.wait_stream(main); with torch.cuda.stream(pf): launch _l2_warm_kernel on the o view, then on the pooler slice; master attention continues on main unchanged. Join with main.wait_stream(pf) immediately before the return (the warm-up is long finished by then; this only closes the fork for graph capture). Only constant prepare-time weights are read; nothing derived from inputs is read or kept. Every other shape: master byte for byte.

## Expected

- `*layer=sliding load=decode_16 *` faster: o (nvjet 64x16 2x1 pair with the unpooler, 26.1 us) and pooler (nvjet 128x16 splitK, 20.0 us) read 42 of their 84 MB from L2: milestone pooler row <= 15 us; -4..-8 us of 230 us, +2-3.5%
- `*layer=sliding load=decode_32 *` faster: same weights, pooler 21.2 us: -4..-8 us of 270 us, +1.5-3%
- `*layer=sliding load=verify_1 *` faster: o 7.1 us + pooler half of the 128x8 splitK pair: -3..-6 us of 205 us, +1.5-3%
- `*layer=full*` flat: untouched: full-layer attention streams 300+ MB of KV and would evict the warm lines
- `*load=decode_1 *` flat: T=1 outside the gate (restack changes those GEMMs)
- `*load=verify_16 *` flat: T=128 outside the gate

## Risk

Bold probe of a lever nothing has tried: each timed replay starts with a cold L2, and on sliding shapes the attention phase leaves HBM nearly idle for ~25 us. Counter-evidence (LESSONS): moe w2-l2-prefetch-decode1 cost 13.8 us and Marlin did not speed up, but that warm-up was serialized before Marlin and Marlin at bs=1 is not bandwidth-bound; here the warm-up is overlapped and the targets are cuBLAS GEMMs at 3.2-3.4 TB/s. Read the o and pooler nvjet rows and the 2 x warm rows; if the GEMM rows do not move, the knob is dead. Overlap: sliding-fused-batched-decode-port shortens the decode_16/32 attention window (less time to hide 42 MB) and small-t-launch-minimal-dense-chain-pdl (in judge) replaces the pooler kernel at T<=32 when ON (then only the o part applies). All three are separate checks on master, so nothing is confounded in the judge today.
