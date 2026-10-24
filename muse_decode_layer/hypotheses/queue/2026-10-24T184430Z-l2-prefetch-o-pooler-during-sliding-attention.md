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
review: proceed
status: queued
review_at: 2026-10-24T185025Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T185025Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed, but only after retargeting the pooler warm-up: since small-t-launch-minimal-dense-chain-pdl landed, master's pooler at T<=32 reads state["dl"]["w_pr"] (a separate bf16 [4608,8192] copy), not dense["pooler"]. Built as written, 25 MB of the warm-up would load a buffer nothing reads and pin it evict_last in L2, so a loss would wrongly mark the knob as dead.

### Justification
The mechanism is plausible. Every timed replay starts with a cold L2. On sliding T<=32 shapes the qk_norm/store/FA3(2048-window, <=8 MB KV)/o_norm_gate phase moves little DRAM traffic. The o GEMM and kernel P (pooler+router, 75.5 MB) right after it are HBM-bound. A 42 MB warm-up fits under the 50 MB L2, and FINDINGS shows those rows are material: dl_pooler_router_kernel is 23.9 us at sliding decode_16, 28.3 at decode_32 and 22.5 at verify_1. The o GEMM shares nvjet rows with the unpooler: 64x16 at 26.5 us at decode_16, 64x32 at 28.6 at decode_32, and 128x8 splitK at 12.8 at verify_1. The hypothesis text is stale, though. It cites the old nvjet 128x16 splitK pooler row (20.0 us) and incumbents of 230/270/205 us. The current numbers are 217.1/247.0/186.5 us, and at T<=32 the pooler is kernel P over dl["w_pr"]. The pooler view must therefore be dl["w_pr"][:1536] (rows 0..1535 are pooler rows in the same order), falling back to dense["pooler"][:1536] only when state["dl"] is None. Expected sizes: o is ~20% of its nvjet row (16.8 of ~84 MB at T=16/32), so warming it saves ~2-3 us. Kernel P is a single wave (216 CTAs at 2/SM) that is bandwidth-bound on total bytes, so removing 25 MB of 75.5 MB from HBM could save ~5-7 us if the lines survive. That is -5..-9 us on the three sliding shapes, ~+2-4% each. The headline effect is only ~0.5-1% (3 of 6 loads, sliding share ~2/3 of each load), inside the +-1.5% CI, so the result must be read from the profile rows (P row, o/unpooler nvjet row, the two warm rows, the FA3 row) and not from the headline. WEAKEST LINK: whether 32 programs can pull 42 MB (needs ~1.7-2 TB/s, i.e. ~55-65 GB/s per SM with plain tl.load) inside a sliding attention phase that is probably shorter than the assumed 25-30 us. If the warm-up is still running when the o GEMM and P start, it competes for HBM and keeps up to 32 SMs busy. P is sized as exactly one wave at 2 CTAs/SM, so leftover warm CTAs can push some P CTAs into a second wave. Under-delivery or a loss is the likely failure mode, and the warm rows' own duration in the profile is the deciding number. A second risk is the line survival budget: 42 MB of evict_last lines, plus 8 MB of concurrent KV traffic, plus o_norm_gate/boundary activations, sits right at the 50 MB capacity.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (landed, the base): changes which buffer the pooler reads at T<=32 (dl["w_pr"] instead of dense["pooler"]), so the change as written warms a dead buffer. Kernel P also already issues 3 weight stages per CTA (~10 MB chip-wide) before griddepcontrol.wait, overlapping part of the weight stream with the predecessor, so the warm-up's marginal gain on P is smaller than against the old nvjet pooler. P's single-wave sizing (216 CTAs at 2/SM) is sensitive to warm CTAs still being resident. If P were absent or OFF (self-check fail, logged as [small-t-dense-chain] OFF), the right target and the size of the gain both change, so the build must log which buffer it warmed.
- sglang-pipeline-parity-router-hilo-m8: the side-stream router/pooler split it introduced no longer runs at T<=32 (superseded by P), so it does not confound inside the gate.
- fa3-hdim128-head-pairing-batched: full layers only (window == 0), outside the sliding gate; it does not confound, and full layers at the same T serve as an in-run control.
- post-expert-norm-triton-small-t and marlin-block-m-cap-16: run after the o and pooler GEMMs and read different weights. The only possible interaction is evict_last lines left in L2 during Marlin/unpooler, which are pure streams that gain little from L2. Not a meaningful confounder.

### Interference and A/B needs
No companion is warranted. The gate is config-only and limited to three sliding shapes. The same-T full-layer shapes and every T=1/T>32 shape are byte-for-byte master in the same run, so they act as built-in controls. Attribution comes from per-kernel profile rows: the P row (pooler part), the o/unpooler nvjet row (the unpooler is untouched, so any delta is o), and the two _l2_warm_kernel rows and their duration against the FA3/o_norm_gate rows. Together these separate the mechanisms. A gain shows up as a shorter P and o row. A loss splits into three readable cases: warm rows longer than the attention phase (HBM/SM contention, a longer FA3 row), P running slower than master (a second wave or contention), or P unchanged (lines evicted). An ab-revert of small-t-launch-minimal-dense-chain-pdl would test the pooler part against a different kernel and buffer that is no longer master's path, which is not useful. An o-only control would only separate o from the pooler, which the rows already do. The headline will probably sit inside the CI either way, so judge the run on the rows, not the headline.

### Open questions
- How long is the sliding attention phase on master now (qk_norm + store_cache + FA3 + o_norm_gate rows at decode_16/32/verify_1)? If it is under ~20 us, consider warming only the o weight plus fewer pooler rows (e.g. 16.8 + ~12 MB) so the warm-up reliably finishes before the o GEMM.
- Does the warm-up grid of 32 programs reach ~2 TB/s? Watch the _l2_warm_kernel row durations. 64 programs with smaller per-program ranges may be needed, at the cost of more SM pressure on FA3.
- Which pooler rows should be warmed for kernel P? All 216 CTAs run in one wave and each streams a K-chunk of 64 rows, so warming rows 0..1535 makes 24 of 72 n-tiles cheap while the rest still set the HBM-bound tail. Total bytes saved is what matters, but check that P's row drops by roughly 25 MB / ~3.3 TB/s ≈ 7 us.
- Overlap with the queued sliding-fused-batched-decode-port (shorter attention window to hide in) and prefork-qkv-o-swapab-t16-t32 (a different o kernel). Whichever lands first changes this one's budget, so re-baseline before stacking.

