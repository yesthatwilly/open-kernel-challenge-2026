# muse_prefill_layer · 2026-10-25T153907Z · submit · qkv-gate-columns-side-stream-under-attention

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T152519Z-qkv-gate-columns-side-stream-under-attention.md` (branch `hyp/muse_prefill_layer/qkv-gate-columns-side-stream-under-attention`).

- **Job:** `b8b8536c4ed74d78` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `90b7f9ef`

## Goal
Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection is one GEMM today (qkv = _gemm(attn_input, dense['qkv_gate']), N = 2304 = q 1024 | k 128 | v 128 | output gate 1024), but its last 1024 columns (the output gate) are read only by state['o_norm_gate'](..., gate) AFTER FlashAttention. Split the projection by weight rows and run the gate part on the side stream under the attention chain. prepare(): state['qkv_w_main'] = weights['qkv_gate'][:1280].contiguous() and state['qkv_w_gate'] = weights['qkv_gate'][1280:].contiguous() (bf16 row slices: a lossless repack; master's dense['qkv_gate'] stays for the OFF path). kernel_fn, when state['qkv_split_ok'].get((tclass, cfg['window'] != 0)) (tclass from tokens only: 'low' T <= 1024, 'high' T <= 4096, 'ft' above; window from config: shape/config branches only): qkv = torch.empty((T, 2304), bf16); torch.mm(attn_input, state['qkv_w_main'].t(), out=qkv[:, :1280]) on the main stream; side = state['side_stream']; side.wait_stream(main); with torch.cuda.stream(side): torch.mm(attn_input, state['qkv_w_gate'].t(), out=qkv[:, 1280:]); the main stream then runs master's code unchanged from FusedQKNorm through flash_attn_with_kvcache (qk_norm, the k/v FP8 casts, store_cache, q FP8 cast, FA3, which read only columns < 1280), then main.wait_stream(side) immediately before state['o_norm_gate'](out..., gate) with gate = qkv[:, 1280:] exactly as today. Outside capture, call attn_input.record_stream(side) and qkv.record_stream(side) as master does for the routing tensors. The out= targets are column slices with row stride 2304 and unit column stride, which ATen passes to cuBLAS as ldc = 2304 without a copy: prepare() must PROVE it once (torch.profiler over one eager call of the split pair: no copy/elementwise kernel, exactly two GEMM kernels, and qkv's data_ptr unchanged), else the path stays OFF. The router later reuses state['side_stream'] in order (it already waits on main), so no new stream is needed; nothing module-global. Everything after o_norm_gate is master's code. PREPARE GATE (cheap, < 15 s, run last): for each key (T, layer) in {753, 3012, 16384} x {sliding window 2048 + rope, full}, build one synthetic Inputs with the scored shape's batch/q_len/ctx (b1 ctx 80000 at 753, b4 ctx 80000 at 3012, b1 ctx 32768 at 16384; page_size 1, random FP8 k/v cache rows, kv_scale 1, scheduler_metadata None as at prefill), and capture two graphs of the attention front only (qkv projection through o_norm_gate, master vs split); 15 interleaved alternations, 256 MiB flush write before each replay, empty-graph baseline subtracted; ON only if split <= 0.97x master for that key; plus a parity check on that case: o_norm_gate output rel L2 <= 2e-3 vs master and the written k/v cache rows byte-identical (the split GEMM's q/k/v columns may differ by accumulation order only if cuBLAS picks a different tile; any FP8 byte difference beyond 0.1% of rows turns the key OFF). Print '[qkv-split] GATE T=<T> <layer> master <us> split <us> ratio <r>: ON/OFF' and a '[qkv-split] PATH CHOSEN' line after rega's and align2's lines. Free the synthetic caches before returning. Mechanism: at T=753 the whole qkv GEMM (nvjet 112x128, 39 us) sits on the critical path before qk_norm and an FA3 call of 25.3 us (sliding) that cannot fill 132 SMs at batch 1; the 1024-column gate GEMM (12.6 GFLOP, ~15-17 us) can run on idle SMs under qk_norm/store/FA3. At T=3012 qkv is 150 us (nvjet 192x144) before a 61.8 us sliding FA3; at T=16384 qkv is ~786 us of the 2533 us 192x192 trio before 306 us (sliding) / 912-8350 us (full) FA3 plus 43 us qk_norm. The same side-stream concurrency already pays for the router under the pooler (profile 5: 172 us of kernel overlap, pooler only ~90 us slower).

## What changed, and where
Edited only `.worktrees/muse_prefill_layer/qkv-gate-columns-side-stream-under-attention/muse_prefill_layer/submission.py` (+318/-8). Base 712ef67f; master 8ae0a9c9 has the same submission.py blob, so no merge was needed.

What changed:
- kernel_fn's attention front moved verbatim into `_attn_chain` (FusedQKNorm through FA3, with an optional `fork` callback right after store_cache) and `_attn_front(data, state, mode)`. `mode None` is master's code (one `_gemm`, one stream). kernel_fn reads `state["qkv_split_ok"].get((_qkv_class(tokens), window != 0))`, a shape/config branch only.
- Split mode `(fork, prio)`: `qkv = torch.empty((T, 2304))`; q|k|v GEMM `torch.mm(x, qkv_w_main.t(), out=qkv[:, :1280])` on main; the gate GEMM into `qkv[:, 1280:]` on `state["side_stream"]`, forked with `side.wait_stream(src)` either from main right after the q|k|v GEMM (fork "qkv") or from the chain stream right after store_cache (fork "store"); the chain runs on `state["hp_stream"]` (priority -1) when prio, else on main; `main.wait_stream(chain)` and `main.wait_stream(side)` come immediately before `o_norm_gate`. Event-based only, graph-capturable.
- prepare(): `_qkv_split_setup` runs after master's whole setup (rega and the align2 re-print), so rega's 150 s first_turn deadline is untouched. It makes qkv_w_main / qkv_w_gate (row slices of dense qkv_gate), creates the priority -1 stream (falls back to no-hp variants if unsupported), and runs the profiler no-copy proof. It then gates each of 3 T classes x {sliding, full} = 6 keys on the real attention front (synthetic scored-shape inputs: b1 ctx 80000 T=753, b4 ctx 80000 T=3012, b1 ctx 32768 T=16384, page 1, random FP8 caches), master vs 4 variants {qkv, store} x {hp, main}, as captured graphs, 15 interleaved alternations, 256 MiB flush, empty-graph baseline subtracted. Prints `[qkv-split] GATE T=.. <layer> master .. split .. ratio ..: ON/OFF` per key and `[qkv-split] PATH CHOSEN` last. Skipped (all OFF) if prepare already used > 240 s. Everything is wrapped so it never raises and leaves master's path.

Adversarial review items:
1. Byte-identical KV parity replaced by tolerance: dequantized K and V rel L2 <= 1e-2 and worst row <= 5e-2 (row norm floored at 0.1 x mean row norm); output (o_norm_gate) rel L2 vs master <= 2e-3 kept; differing-row fractions printed for information. The K/V rows master wrote are zeroed before the split run so a missed write shows.
2. FA3 starvation: both fork points and the priority -1 chain stream are timed; best variant per key wins if <= 0.98x master and parity passes, else master. Whether a captured graph keeps the stream priority is not observable in-process; this is logged and the gate decides.
3. attn_input.record_stream dropped (and qkv.record_stream too: every side stream joins main before the function returns, so none is needed).
4. Fork/join event-based; side stream and hp stream both join before o_norm_gate; the router later reuses side_stream in order as before.
5. Gate per (T class, layer) on the real subgraph, cold L2, threshold 0.98 (spec said 0.97; the review says 0.98, applied), runs last.
6. Profiler proof runs in prepare only (no copy/elementwise/memset kernel, 2..4 GEMM-class kernels, data_ptr unchanged), else all OFF.

Deviations / open questions:
- Gate threshold 0.98 (review) not 0.97 (spec).
- Full-layer ft is gated once at ctx 32768, not keyed by ctx (the review only suggested it). First-turn full rows at ctx 0 / 65536 share that decision.
- Profiler proof allows up to 4 GEMM-class kernels (a split-K reduce counts as GEMM-class) rather than exactly 2.
- Landing order vs fa3-full-contiguous-kv-triton-gather-gated (same kernel_fn span): not decided here; expect a textual merge conflict, and the full-layer keys would need re-gating.
- Nothing run (no GPU on this node): the profiler-event API, `torch.cuda.Stream(priority=-1)` and `torch.mm(out=<column slice>)` are the compile/run risks; each fails to master.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no syncs/prints/allocation on values in kernel_fn path; `_attn_front` only does shape-based branching, torch.empty by shape, event-based stream fork/join (checked in diff).
- [x] Nothing derived from `data` is cached: all per-call tensors are locals; state holds only weights views, streams and the ON/OFF dict.
- [x] Weights: qkv_w_main / qkv_w_gate are bf16 row slices of the bf16 qkv_gate (lossless).
- [x] Activations: qkv stays BF16, q to FP8 as master, KV cache FP8.
- [x] Attention sees every key: FA3 call unchanged (same window / context args).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (the synthetic inputs are built locally).
- [x] All work inside the timed call; no output caching or replay.
- [x] No module-global tensors/buffers: streams, weights and flags live in `state`; module constants are scalars.
- [x] No try/except or prints in kernel_fn, `_attn_front`, `_attn_chain`.

okc_lint: `OK with warnings: submission.py` (only the existing host-sync-in-prepare warnings, none new in kernel_fn's call graph). py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Risk medium: sign of the overlap is not certain (two kernels time-sharing SMs can lose to one well-tiled GEMM: N=1280 and N=1024 tiles quantize worse than N=2304), hence the per-(T, layer) prepare gate on the real attention-front subgraph. The out= column-slice trick must be verified not to copy (profiler proof in prepare), else OFF. Read the result on the per-shape rows and the nvjet rows (expect two different GEMM rows in place of the qkv one, and FA3/qk_norm rows unchanged): a lower qkv-row sum alone is not the claim, the shape time is. Does not touch the router, pooler or expert paths; disjoint from pre-expert-norm-fused-rega-gather, so the two can be stacked later.

## Result (after the run)
- job `b8b8536c4ed74d78` (submit), status done, passed True, tests 5/5
- speedup 1.4512x, 95% CI [1.4294, 1.4730], score 266955.64779338986 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4586709689454935, "ci": [1.4367909044113112, 1.4805510334796759], "job": "1cc92a43b8be40f8"}; base result: muse_prefill_layer/hypotheses/done/neutral/align-two-phase-all-t-rega.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/qkv-gate-columns-side-stream-under-attention.result.json`, `muse_prefill_layer/hypotheses/done/neutral/qkv-gate-columns-side-stream-under-attention.patch`; branch `hyp/muse_prefill_layer/qkv-gate-columns-side-stream-under-attention`
