---
problem: muse_prefill_layer
slug: qkv-gate-columns-side-stream-under-attention
created: 2026-10-25T152519Z
base: master
area: qkv GEMM (output-gate columns split onto the side stream, overlapping qk_norm + cache store + FA3)
change: >
  Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection is
  one GEMM today (qkv = _gemm(attn_input, dense['qkv_gate']), N = 2304 = q 1024 | k 128 | v 128 |
  output gate 1024), but its last 1024 columns (the output gate) are read only by
  state['o_norm_gate'](..., gate) AFTER FlashAttention. Split the projection by weight rows and
  run the gate part on the side stream under the attention chain. prepare(): state['qkv_w_main'] =
  weights['qkv_gate'][:1280].contiguous() and state['qkv_w_gate'] =
  weights['qkv_gate'][1280:].contiguous() (bf16 row slices: a lossless repack; master's
  dense['qkv_gate'] stays for the OFF path). kernel_fn, when state['qkv_split_ok'].get((tclass,
  cfg['window'] != 0)) (tclass from tokens only: 'low' T <= 1024, 'high' T <= 4096, 'ft' above;
  window from config: shape/config branches only): qkv = torch.empty((T, 2304), bf16);
  torch.mm(attn_input, state['qkv_w_main'].t(), out=qkv[:, :1280]) on the main stream; side =
  state['side_stream']; side.wait_stream(main); with torch.cuda.stream(side): torch.mm(attn_input,
  state['qkv_w_gate'].t(), out=qkv[:, 1280:]); the main stream then runs master's code unchanged
  from FusedQKNorm through flash_attn_with_kvcache (qk_norm, the k/v FP8 casts, store_cache, q FP8
  cast, FA3, which read only columns < 1280), then main.wait_stream(side) immediately before
  state['o_norm_gate'](out..., gate) with gate = qkv[:, 1280:] exactly as today. Outside capture,
  call attn_input.record_stream(side) and qkv.record_stream(side) as master does for the routing
  tensors. The out= targets are column slices with row stride 2304 and unit column stride, which
  ATen passes to cuBLAS as ldc = 2304 without a copy: prepare() must PROVE it once (torch.profiler
  over one eager call of the split pair: no copy/elementwise kernel, exactly two GEMM kernels, and
  qkv's data_ptr unchanged), else the path stays OFF. The router later reuses state['side_stream']
  in order (it already waits on main), so no new stream is needed; nothing module-global.
  Everything after o_norm_gate is master's code. PREPARE GATE (cheap, < 15 s, run last): for each
  key (T, layer) in {753, 3012, 16384} x {sliding window 2048 + rope, full}, build one synthetic
  Inputs with the scored shape's batch/q_len/ctx (b1 ctx 80000 at 753, b4 ctx 80000 at 3012, b1
  ctx 32768 at 16384; page_size 1, random FP8 k/v cache rows, kv_scale 1, scheduler_metadata None
  as at prefill), and capture two graphs of the attention front only (qkv projection through
  o_norm_gate, master vs split); 15 interleaved alternations, 256 MiB flush write before each
  replay, empty-graph baseline subtracted; ON only if split <= 0.97x master for that key; plus a
  parity check on that case: o_norm_gate output rel L2 <= 2e-3 vs master and the written k/v cache
  rows byte-identical (the split GEMM's q/k/v columns may differ by accumulation order only if
  cuBLAS picks a different tile; any FP8 byte difference beyond 0.1% of rows turns the key OFF).
  Print '[qkv-split] GATE T=<T> <layer> master <us> split <us> ratio <r>: ON/OFF' and a
  '[qkv-split] PATH CHOSEN' line after rega's and align2's lines. Free the synthetic caches before
  returning. Mechanism: at T=753 the whole qkv GEMM (nvjet 112x128, 39 us) sits on the critical
  path before qk_norm and an FA3 call of 25.3 us (sliding) that cannot fill 132 SMs at batch 1;
  the 1024-column gate GEMM (12.6 GFLOP, ~15-17 us) can run on idle SMs under qk_norm/store/FA3.
  At T=3012 qkv is 150 us (nvjet 192x144) before a 61.8 us sliding FA3; at T=16384 qkv is ~786 us
  of the 2533 us 192x192 trio before 306 us (sliding) / 912-8350 us (full) FA3 plus 43 us qk_norm.
  The same side-stream concurrency already pays for the router under the pooler (profile 5: 172 us
  of kernel overlap, pooler only ~90 us slower).
expected:
  - shapes: "*layer=sliding load=turn_1*"
    direction: faster
    why: "the 1024-column gate GEMM (~15-17 us) leaves the critical path and overlaps qk_norm + store + a 25 us FA3 that leaves SMs idle at batch 1: -8 to -15 us of 747 (1-2%)"
  - shapes: "*layer=sliding load=turn_4*"
    direction: faster
    why: "gate GEMM ~65 us (of the 150 us qkv) overlaps qk_norm + store + 61.8 us FA3: -25 to -50 us of 2151 (1.2-2.3%)"
  - shapes: "*load=first_turn*"
    direction: faster
    why: gate GEMM ~350 us overlaps qk_norm (30-43 us), casts, store and FA3 (306 us sliding, 0.9-8.4 ms full); if overlap behaves like the router under the pooler, -100 to -200 us of 11120 (sliding, ~1-1.8%), smaller relative on full rows
  - shapes: "*layer=full load=turn*"
    direction: flat
    why: full-layer FA3 (417 us turn_1, 1643 us turn_4) may already fill the SMs, so the gate GEMM only time-shares; the per-key gate keeps master where split is not >= 3% faster
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-gate-columns-side-stream-under-attention

## Change

Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection is one GEMM today (qkv = _gemm(attn_input, dense['qkv_gate']), N = 2304 = q 1024 | k 128 | v 128 | output gate 1024), but its last 1024 columns (the output gate) are read only by state['o_norm_gate'](..., gate) AFTER FlashAttention. Split the projection by weight rows and run the gate part on the side stream under the attention chain. prepare(): state['qkv_w_main'] = weights['qkv_gate'][:1280].contiguous() and state['qkv_w_gate'] = weights['qkv_gate'][1280:].contiguous() (bf16 row slices: a lossless repack; master's dense['qkv_gate'] stays for the OFF path). kernel_fn, when state['qkv_split_ok'].get((tclass, cfg['window'] != 0)) (tclass from tokens only: 'low' T <= 1024, 'high' T <= 4096, 'ft' above; window from config: shape/config branches only): qkv = torch.empty((T, 2304), bf16); torch.mm(attn_input, state['qkv_w_main'].t(), out=qkv[:, :1280]) on the main stream; side = state['side_stream']; side.wait_stream(main); with torch.cuda.stream(side): torch.mm(attn_input, state['qkv_w_gate'].t(), out=qkv[:, 1280:]); the main stream then runs master's code unchanged from FusedQKNorm through flash_attn_with_kvcache (qk_norm, the k/v FP8 casts, store_cache, q FP8 cast, FA3, which read only columns < 1280), then main.wait_stream(side) immediately before state['o_norm_gate'](out..., gate) with gate = qkv[:, 1280:] exactly as today. Outside capture, call attn_input.record_stream(side) and qkv.record_stream(side) as master does for the routing tensors. The out= targets are column slices with row stride 2304 and unit column stride, which ATen passes to cuBLAS as ldc = 2304 without a copy: prepare() must PROVE it once (torch.profiler over one eager call of the split pair: no copy/elementwise kernel, exactly two GEMM kernels, and qkv's data_ptr unchanged), else the path stays OFF. The router later reuses state['side_stream'] in order (it already waits on main), so no new stream is needed; nothing module-global. Everything after o_norm_gate is master's code. PREPARE GATE (cheap, < 15 s, run last): for each key (T, layer) in {753, 3012, 16384} x {sliding window 2048 + rope, full}, build one synthetic Inputs with the scored shape's batch/q_len/ctx (b1 ctx 80000 at 753, b4 ctx 80000 at 3012, b1 ctx 32768 at 16384; page_size 1, random FP8 k/v cache rows, kv_scale 1, scheduler_metadata None as at prefill), and capture two graphs of the attention front only (qkv projection through o_norm_gate, master vs split); 15 interleaved alternations, 256 MiB flush write before each replay, empty-graph baseline subtracted; ON only if split <= 0.97x master for that key; plus a parity check on that case: o_norm_gate output rel L2 <= 2e-3 vs master and the written k/v cache rows byte-identical (the split GEMM's q/k/v columns may differ by accumulation order only if cuBLAS picks a different tile; any FP8 byte difference beyond 0.1% of rows turns the key OFF). Print '[qkv-split] GATE T=<T> <layer> master <us> split <us> ratio <r>: ON/OFF' and a '[qkv-split] PATH CHOSEN' line after rega's and align2's lines. Free the synthetic caches before returning. Mechanism: at T=753 the whole qkv GEMM (nvjet 112x128, 39 us) sits on the critical path before qk_norm and an FA3 call of 25.3 us (sliding) that cannot fill 132 SMs at batch 1; the 1024-column gate GEMM (12.6 GFLOP, ~15-17 us) can run on idle SMs under qk_norm/store/FA3. At T=3012 qkv is 150 us (nvjet 192x144) before a 61.8 us sliding FA3; at T=16384 qkv is ~786 us of the 2533 us 192x192 trio before 306 us (sliding) / 912-8350 us (full) FA3 plus 43 us qk_norm. The same side-stream concurrency already pays for the router under the pooler (profile 5: 172 us of kernel overlap, pooler only ~90 us slower).

## Expected

- `*layer=sliding load=turn_1*` faster: the 1024-column gate GEMM (~15-17 us) leaves the critical path and overlaps qk_norm + store + a 25 us FA3 that leaves SMs idle at batch 1: -8 to -15 us of 747 (1-2%)
- `*layer=sliding load=turn_4*` faster: gate GEMM ~65 us (of the 150 us qkv) overlaps qk_norm + store + 61.8 us FA3: -25 to -50 us of 2151 (1.2-2.3%)
- `*load=first_turn*` faster: gate GEMM ~350 us overlaps qk_norm (30-43 us), casts, store and FA3 (306 us sliding, 0.9-8.4 ms full); if overlap behaves like the router under the pooler, -100 to -200 us of 11120 (sliding, ~1-1.8%), smaller relative on full rows
- `*layer=full load=turn*` flat: full-layer FA3 (417 us turn_1, 1643 us turn_4) may already fill the SMs, so the gate GEMM only time-shares; the per-key gate keeps master where split is not >= 3% faster

## Risk

Risk medium: sign of the overlap is not certain (two kernels time-sharing SMs can lose to one well-tiled GEMM: N=1280 and N=1024 tiles quantize worse than N=2304), hence the per-(T, layer) prepare gate on the real attention-front subgraph. The out= column-slice trick must be verified not to copy (profiler proof in prepare), else OFF. Read the result on the per-shape rows and the nvjet rows (expect two different GEMM rows in place of the qkv one, and FA3/qk_norm rows unchanged): a lower qkv-row sum alone is not the claim, the shape time is. Does not touch the router, pooler or expert paths; disjoint from pre-expert-norm-fused-rega-gather, so the two can be stacked later.
