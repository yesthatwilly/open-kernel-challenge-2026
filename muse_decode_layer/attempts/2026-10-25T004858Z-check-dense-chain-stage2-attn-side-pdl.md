# muse_decode_layer · 2026-10-25T004858Z · check · dense-chain-stage2-attn-side-pdl

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T224424Z-dense-chain-stage2-attn-side-pdl.md` (branch `hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl`).

- **Job:** `b31b8453208a471c` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `605685b`

## Goal
Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["dl"] is not None and the new state["dl2_ok"] flag; else master byte for byte (T=128/256 control). Add to the existing _DL_CUDA TU (rename module muse_decode_dense_chain_v2) two kernels built from the dl_pooler_router_kernel<NF> body (same swap-AB mma.sync m16n8k16, 64 weight rows per CTA = exactly one head, 4-stage cp.async ring, fixed-order ticket split-K reduce, PDL launch via cudaLaunchKernelEx with cp.async of weight stages 0-2 BEFORE griddepcontrol.wait and everything that reads activations or writes global AFTER it). KERNEL Q dl_qkv_kernel<NF>: W = dense qkv_gate bf16 [2304, 8192] as is (36 head tiles x SPLIT 6 = 216 CTAs). The CTA that draws the last ticket of a tile owns the 64 fp32 outputs per token and does the epilogue by tile index: tiles 0-15 (q heads): per-head RMSNorm exactly as FusedQKNorm does (read sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py for weight/eps, eps RMS_EPS), RoPE when cfg rope_theta (neox layout and cos_sin from state["cos_sin"] at data positions, exactly as FusedQKNorm applies it), cast to FP8 E4M3 and store to q_attn: unpaired [T,16,64], or when master paired predicate holds (window==0 and batch>1 and hkv%2==0) into the own half of the zero-padded [T,16,128] row (write 64 zeros to the other half); tiles 16-17 (k heads): same norm and RoPE, divide by k_scale (device scalar, only when cfg kv_scale != 1.0, passed as a flag), FP8 cast, write bytes to k_cache row cache_loc[t] head h; tiles 18-19 (v): divide by v_scale when scaled, FP8 cast, store at cache_loc; tiles 20-35 (gate): bf16 to gate [T,1024]. FA3 call unchanged (same q_attn/k_attn/v_attn/metadata logic). KERNEL O dl_o_kernel<NF>: W = dense o bf16 [8192, 1024], 128 row tiles x SPLIT 2 (K halves of 512 = 8 heads) = 256 CTAs at 3/SM (one wave); prologue after griddepcontrol.wait loads FA3 out (bf16, unpaired [T,16,64] or paired [T,16,128] taking its own half) for its 8 heads, per-head scaleless RMSNorm eps RMS_EPS times sigmoid(gate) in fp32 exactly as FusedRMSNormGate, writes the bf16 x tile into the smem x ring (all K of the half fits: 32 x 512 x 2 B = 32 KB), then the weight GEMV; epilogue writes attn_output bf16 [T,8192]. kernel_fn on the gate: qkv = None; Q replaces _gemm(attn_input, qkv_gate), qk_norm, the k/v divide, both .to(FP8), store_cache and the q cast / pad copies; O replaces the unpad copies, o_norm_gate and _gemm(o, dense o). attn_boundary, P, R, experts, unpooler, ffn_boundary unchanged. PREPARE self-check (synthetic, T in 1, 8, 9, 16, 32; sliding+rope and full; paired and unpaired; kv_scale 1 and 0.25): q bytes vs master path rel L2 <= 1e-2 (decode fp8 back to f32; do not require byte identity, LESSONS: Triton sin/cos), cache rows rel L2 <= 1e-2 at cache_loc and untouched elsewhere, attn_output rel L2 <= 5e-3, plus a CUDA-graph capture+replay of the whole gated chain; any failure sets dl2_ok False (master attention side, stage-1 FFN kept). Print [dense-chain-s2] ON|OFF pdl=.. max errs, and Q/O CUDA-event times pdl 0/1 for T=1 and 32.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (not run, no GPU here). The CUDA module is renamed `muse_decode_dense_chain_v2` and gains `-lineinfo` (flags printed at build and in the stage-2 log; no wgmma/setmaxnreg, so plain `-arch=sm_90a` as stage 1 ran).

What changed:
- Two kernels added to the `_DL_CUDA` TU, copied from the P body (swap-AB `mma.sync m16n8k16`, 64 weight rows per CTA, cp.async.cg, ldmatrix.x4, PDL weight prefetch before `griddepcontrol.wait`, fixed-order ticket split-K, in-kernel counter reset).
  - Q `dl_qkv_kernel<NF>`: 36 head tiles x 6 K-splits = 216 CTAs. The last-ticket CTA per tile does the epilogue, one warp per token, lane = element pair, so a lane owns an interleaved RoPE pair. Tiles 0-15: scaleless RMSNorm (eps 1e-5), interleaved RoPE from `state["cos_sin"]` at `positions`, bf16 round, E4M3 into `q_attn`, unpaired [T,16,64] or paired [T,16,128] (own half plus 64 zero bytes, so no zeros fill). Tiles 16-17: same, then divide by k_scale when `kv_scale != 1`, bf16 round, E4M3 to the `cache_loc` row. Tiles 18-19: v, same divide, E4M3 store. Tiles 20-35: bf16 gate [T,1024]. mode 1 (raw) writes the bf16 [T,2304] projection.
  - O `dl_o_kernel<NF>`: 128 row tiles x 2 K-halves = 256 CTAs, all four 16 KB weight K-blocks of the CTA resident in smem (cp.async before the PDL wait). Prologue 1 builds the bf16 x tile in smem from FA3's output (unpaired, or paired taking the head's own half (h>>3)&1) and the gate: per-head scaleless RMSNorm x sigmoid(gate) in fp32 with one bf16 rounding. Prologue 0 (raw) copies the already normed-and-gated [T,1024] input. Epilogue writes attn_output bf16 [T,8192].
- `kernel_fn`: master's attention side moved verbatim into `_master_attention`. `kernel_fn` calls `_dl2_attention` (Q, then the unchanged FA3 call, then O) when `state["dl2_ok"]` and `state["dl"]` are set and `_dl2_applies(cfg)` (config only); otherwise `_master_attention`. Everything after the attention (boundary, P, R, experts, unpooler, ffn_boundary) is untouched.
- prepare(): `_dl2_setup` allocates the ws/counters, prints occupancy, runs the self-check (pdl as stage 1 chose, then 0) and the Q/O timing, and prints `[dense-chain-s2] ON|OFF pdl=... max errs ...` plus three `PATH CHOSEN` lines.

Adversarial-review fixes applied (the review wins over the spec):
- (a) Sliding decode_1 carve-out: `DL2_SLIDING1 = "raw"` runs Q in raw mode (bf16 qkv [T,2304]) into master's `_sliding_forward` and O without the prologue; `"nvjet"` keeps master's whole path there. Config-only, logged. Expected gain at that shape is therefore only the qkv and o GEMM rows, about 5-7 us (review).
- (b) New `DL2_MAX_TOKENS = 32`; `DL_MAX_TOKENS` is not used for gating (ws is sized with the new constant).
- (c) Self-check case batch 4, q_len 8, sliding, kv_scale 0.015625 (T=32).
- (d) Instruction forms limited to cp.async.cg, ldmatrix.x4, mma.sync m16n8k16 bf16; no createpolicy / cache_hint. FP8 encode is `__nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE, __NV_E4M3)` on values already rounded through bf16 (including after the k/v divide); the satfinite note (clamps at 448 where torch's e4m3fn cast gives NaN) is logged. The self-check scales x down so `v / kv_scale` stays below 448.
- (e) Self-check prints before each launch and syncs and prints after it, separately for Q, Q-raw, O and O-raw and the graph chain; Q and O event times at pdl 0 and 1 for T=1 and T=32 are printed.
- (f) Gates are config only; no host sync in `kernel_fn`; counters (`cnt_q`, `cnt_o`) reset in-kernel and are checked back at 0 after the self-check; FA3 attends every key.

Deviations from the spec:
- RoPE is the interleaved (GPT-J) layout (pairs 2i, 2i+1), as reference.py and `_norm_rope` define it, not the neox layout the spec text says; cos/sin come from `state["cos_sin"]` (cos in columns 0-31, sin in 32-63).
- O keeps all four K-blocks of its weight slab resident (64 KB + up to 32 KB x tile, 2 CTAs/SM, 256 CTAs fit one wave) instead of the spec's 3 CTAs/SM ring; occupancy is printed.
- Q's gate output goes to a separate [T,1024] bf16 tensor that O reads, not a slice of a [T,2304] qkv.
- The graph check captures Q, O, Q-raw and O-raw with a stand-in FA3 output, not the real FA3 call (the synthetic inputs have no page table).
- FusedQKNorm / FusedRMSNormGate sources are not on this node, so their bf16 rounding points are inferred from master's tensor dtypes; the self-check compares against the live modules (q and cache rel L2 <= 1e-2, gate 1e-2, attention output <= 5e-3).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in `kernel_fn`'s call graph (`_dl2_attention`, `_dl2_q`, `_dl2_o`, `_master_attention`); the only synchronizes are in prepare()-only helpers (lint reports them as outside kernel_fn). Allocation is by shape only; no value-dependent control flow (gates read `config`). The self-check captures and replays Q/O in a CUDA graph.
- [x] Nothing derived from `data` is cached across calls: ws/counters are prepare() buffers that hold no data (every launch leaves the counters at 0).
- [x] Weights: no new weight copies at all; Q and O read `state["dense"]["qkv_gate"]` and `["o"]` as BF16.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (E4M3 written through bf16); queries E4M3, no coarser.
- [x] Attention sees every key: FA3 call is unchanged (same q_attn / cache / metadata logic); the sliding decode_1 shape keeps `_sliding_forward`.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (git diff grepped for subprocess, multiprocessing, os.fork: none).
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Only instruction forms already proven on the judge (cp.async.cg, ldmatrix.x4, mma.sync m16n8k16 bf16); no createpolicy/cache_hint; no sm_90a-only PTX (no TORCH_CUDA_ARCH_LIST change needed); `-lineinfo` added.
- [x] n/a: bounded mbarrier/spin waits: the new kernels use no mbarrier or spin wait (cp.async groups, `__syncthreads`, a single atomic ticket).
- [x] Any build error, self-check miss or exception lands on master's attention side (stage-1 FFN kept) and never raises; loud `PATH CHOSEN` lines per shape class.

okc_lint output:

```
OK with warnings: muse_decode_layer/submission.py
(warnings only: host-sync lines outside kernel_fn's call graph in prepare(); dtype warnings for the self-check's synthetic uint8 -> FP8 cache init at _dl2_make_data)
```

`python3 -I -m py_compile` passed; `__pycache__` removed.

Uncertainties: never compiled or run (no nvcc here); a first-launch fault would poison the context, so the first run should be an `okc check`.

## Risks / what could fail
high
High: Q epilogue must match FusedQKNorm (weights, RoPE layout) and the cache byte layout; a wrong RoPE fails tests, so the self-check gates it. LESSONS: small-kernel fusion alone has been flat, so the gain must come from the dense streams plus node count together; judge on replaced rows. Overlaps queued fused-preattention-paired-pad-restack and held prefork-qkv-o-swapab-t16-t32 (same qkv/o rows at T=16/32 with Triton); if either lands first, compare against master at claim. The running sliding-fused-batched-decode-port replaces the sliding batch>1 attention chain: if it lands first, keep its branch on sliding batch>1 and gate this to the remaining shapes.

## Result (after the run)
- job `b31b8453208a471c` (check), status done, passed True, tests 13/13
- speedup 1.2253x, 95% CI [1.2069, 1.2437], score 18461.107156220427 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/dense-chain-stage2-attn-side-pdl.result.json`, `muse_decode_layer/hypotheses/done/regression/dense-chain-stage2-attn-side-pdl.patch`; branch `hyp/muse_decode_layer/dense-chain-stage2-attn-side-pdl`
